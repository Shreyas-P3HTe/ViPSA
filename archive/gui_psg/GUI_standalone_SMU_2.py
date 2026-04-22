# -*- coding: utf-8 -*-
"""
Refactored on Tue Aug 12 2025 (SGT)

Goal: Rewire Standalone Measurement GUI to use Main4.Vipsa_Methods for
- running measurements (IV & Pulse)
- saving data
- plotting figures

Notes:
- We avoid connecting cameras/stages/lights. This is a manual-probing GUI.
- We instantiate Vipsa_Methods and only attach a KeysightSMU instance to it.
- Alignment/approach are disabled (align=False, approach=False) so no mechanics are touched.
- List generation remains local (simple helper inside this file). You can swap
  to Datahandling.Listmaker later if you prefer.

Dependencies expected on PYTHONPATH:
- Source_Measure_Unit.KeysightSMU
- Main4.Vipsa_Methods
- (Main4 internally uses Datahandling for saving/plotting)
"""

import os
import csv
import time
import threading
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import pyvisa

from typing import List, Tuple

# External backends
from Source_Measure_Unit import Keithley707B, KeithleySMU, KeysightSMU
from Main4 import Vipsa_Methods

# ------------------------------
# Config
# ------------------------------
DEFAULT_SAVE_DIRECTORY = "./manual_probe_data"
DEFAULT_SWEEP_PATH = ""
DEFAULT_PULSE_PATH = ""

# ------------------------------
# Minimal Listmaker (time-voltage CSV helpers)
# ------------------------------
class ListmakerHelpers:
    @staticmethod
    def generate_iv_times_voltages(forward_v: float, reset_v: float, step_v: float,
                                   step_delay: float, forming: bool, forming_v: float,
                                   cycles: int) -> Tuple[List[float], List[float]]:
        t, v = [], []
        cur_t = 0.0

        def up_down(vmax: float):
            nonlocal cur_t
            vv = 0.0
            while vv <= vmax:
                v.append(vv); t.append(cur_t); cur_t += step_delay; vv += step_v
            vv -= step_v
            while vv >= 0:
                v.append(vv); t.append(cur_t); cur_t += step_delay; vv -= step_v

        def down_up(vmin: float):
            nonlocal cur_t
            vv = 0.0
            while vv >= vmin:
                v.append(vv); t.append(cur_t); cur_t += step_delay; vv -= step_v
            vv += step_v
            while vv <= 0:
                v.append(vv); t.append(cur_t); cur_t += step_delay; vv += step_v

        if forming and forming_v is not None:
            up_down(forming_v)
            down_up(reset_v)

        n = cycles if not forming else max(cycles - 1, 0)
        for _ in range(n):
            up_down(forward_v)
            down_up(reset_v)

        if v and v[-1] != 0:
            v.append(0.0); t.append(cur_t)
        return t, v

    @staticmethod
    def generate_pulse_times_voltages(set_n, set_v, set_w, read_n, read_v, read_w,
                                      erase_n, erase_v, erase_w, cycles) -> Tuple[List[float], List[float]]:
        t, v = [], []
        cur_t = 0.0
        for _ in range(int(cycles)):
            for _ in range(int(set_n)):
                v.append(set_v); t.append(cur_t); cur_t += set_w
                v.append(0.0); t.append(cur_t)
            for _ in range(int(read_n)):
                v.append(read_v); t.append(cur_t); cur_t += read_w
                v.append(0.0); t.append(cur_t)
            for _ in range(int(erase_n)):
                v.append(erase_v); t.append(cur_t); cur_t += erase_w
                v.append(0.0); t.append(cur_t)
            for _ in range(int(read_n)):
                v.append(read_v); t.append(cur_t); cur_t += read_w
                v.append(0.0); t.append(cur_t)
        return t, v

    @staticmethod
    def save_to_csv(times: List[float], volts: List[float], filepath: str) -> bool:
        try:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Time (s)", "Voltage (V)"])
                for ti, vi in zip(times, volts):
                    w.writerow([f"{ti:.6f}", f"{vi:.6f}"])
            return True
        except Exception as e:
            print("Error saving CSV:", e)
            return False

    @staticmethod
    def plot_preview(times: List[float], volts: List[float], title: str):
        plt.figure(figsize=(10, 5))
        plt.plot(times, volts, marker=".", linewidth=1)
        plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)"); plt.title(title)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout(); plt.show()


# ------------------------------
# GUI
# ------------------------------
class StandaloneTesterGUI:
    def __init__(self):
        sg.theme("Reddit")
        self.vipsa = Vipsa_Methods()
        self.smu = None
        self.smus = {}
        self.switch = None
        self.discovered_instruments = {}
        self.active_smu_label = None
        self.is_smu_connected = False

        self.generated_iv = None   # (times, volts)
        self.generated_pulse = None

        self.window = self._create_window()

    # ---- Layout
    def _create_window(self):
        list_iv = [
            [sg.Text("— I–V Sweep List —", font=("", 11, "bold"))],
            [sg.Text("Forward V"), sg.Input("1.5", key="IV_FWD", size=(8,1)),
             sg.Text("Reset V"), sg.Input("-1.5", key="IV_RST", size=(8,1))],
            [sg.Text("Step V"), sg.Input("0.05", key="IV_STEP", size=(8,1)),
             sg.Text("Delay (s)"), sg.Input("0.001", key="IV_DELAY", size=(8,1))],
            [sg.Text("Cycles"), sg.Input("2", key="IV_CYC", size=(8,1))],
            [sg.Checkbox("Forming?", key="IV_FORM", enable_events=True),
             sg.Text("Form V"), sg.Input("", key="IV_FORM_V", size=(8,1), disabled=True)],
            [sg.Button("Preview IV"), sg.Button("Save IV CSV")],
        ]

        list_pulse = [
            [sg.Text("— Pulse List —", font=("", 11, "bold"))],
            [sg.Text("Set (#,V,W)"), sg.Input("1", key="PS_N", size=(5,1)), sg.Input("1.5", key="PS_V", size=(6,1)), sg.Input("0.001", key="PS_W", size=(8,1))],
            [sg.Text("Read (#,V,W)"), sg.Input("1", key="PR_N", size=(5,1)), sg.Input("0.1", key="PR_V", size=(6,1)), sg.Input("0.001", key="PR_W", size=(8,1))],
            [sg.Text("Erase (#,V,W)"), sg.Input("1", key="PE_N", size=(5,1)), sg.Input("-1.5", key="PE_V", size=(6,1)), sg.Input("0.001", key="PE_W", size=(8,1))],
            [sg.Text("Cycles"), sg.Input("10", key="P_CYC", size=(6,1))],
            [sg.Button("Preview Pulse"), sg.Button("Save Pulse CSV")],
        ]

        tab_list = [[sg.Column(list_iv)], [sg.HSeparator()], [sg.Column(list_pulse)]]

        manual = [
            [sg.Text("— SMU —", font=("", 11, "bold"))],
            [sg.Button("Connect All Equipment"), sg.Button("Disconnect All Equipment"), sg.Text("Status: Disconnected", key="SMU_STATUS", text_color="red")],
            [sg.Text("Active SMU"), sg.Combo(["Keithley", "Keysight"], default_value="Keithley", key="ACTIVE_SMU", readonly=True, size=(12,1)), sg.Button("Switch Instrument")],
            [sg.HSeparator()],
            [sg.Text("Sample"), sg.Input("SampleX", key="SAMPLE", size=(12,1)),
             sg.Text("Device"), sg.Input("Dev1", key="DEVICE", size=(8,1))],
            [sg.Text("Save Folder"), sg.Input(DEFAULT_SAVE_DIRECTORY, key="SAVE_DIR", size=(40,1)), sg.FolderBrowse()],
            [sg.HSeparator()],
            [sg.Text("— Run DCIV from CSV —", font=("", 11, "bold"))],
            [sg.Text("Active SMU"), sg.Combo(["Keithley", "Keysight"], default_value="Keithley", key="IV_SMU", readonly=True, size=(12,1), disabled=True)],
            [sg.Text("Sweep CSV"), sg.Input(DEFAULT_SWEEP_PATH, key="SWEEP_PATH", size=(40,1)), sg.FileBrowse()],
            [sg.Text("Pos Comp (A)"), sg.Input("0.001", key="POS", size=(10,1)), sg.Text("Neg Comp (A)"), sg.Input("0.01", key="NEG", size=(10,1))],
            [sg.Text("Delay (s, opt)"), sg.Input("0.0001", key="SW_DELAY", size=(10,1)), sg.Text("Plot?"), sg.Checkbox("", key="PLOT_IV", default=True), sg.Text("Timeout (s)"), sg.Input("5", key="IV_TO", size=(4,1))],
            [sg.Button("Run DCIV")],
            [sg.HSeparator()],
            [sg.Text("— Run Pulsed from CSV —", font=("", 11, "bold"))],
            [sg.Text("Active SMU"), sg.Combo(["Keysight", "Keithley"], default_value="Keysight", key="PULSE_SMU", readonly=True, size=(12,1), disabled=True)],
            [sg.Text("Pulse CSV"), sg.Input(DEFAULT_PULSE_PATH, key="PULSE_PATH", size=(40,1)), sg.FileBrowse()],
            [sg.Text("Compliance (A)"), sg.Input("0.01", key="COMP", size=(10,1)), sg.Text("Width (s, opt)"), sg.Input("", key="PWIDTH", size=(10,1)), sg.Text("Plot?"), sg.Checkbox("", key="PLOT_P", default=True), sg.Text("Timeout (s)"), sg.Input("5", key="P_TO", size=(4,1))],
            [sg.Button("Run Pulse")],
        ]

        proto_editor = [
            [sg.Text("Test Type")],
            [sg.Combo(["DCIV from File", "Pulse from File"], key="P_TYPE", readonly=True)],
            [sg.Button("Add Step"), sg.Button("Remove Step"), sg.Button("Clear")],
        ]
        proto_view = [
            [sg.Text("Protocol Steps")],
            [sg.Listbox(values=[], key="P_LIST", size=(60,10))],
            [sg.Text("Sample"), sg.Input("ProtoSample", key="P_SAMPLE", size=(12,1)), sg.Text("Device"), sg.Input("ProtoDev", key="P_DEVICE", size=(10,1))],
            [sg.Text("Save Folder"), sg.Input(DEFAULT_SAVE_DIRECTORY, key="P_SAVE", size=(40,1)), sg.FolderBrowse()],
            [sg.Text("Plot?"), sg.Checkbox("", key="P_PLOT", default=False), sg.Text("Timeout (s)"), sg.Input("3", key="P_TO_MS", size=(4,1))],
            [sg.Button("Run Protocol")],
        ]

        logs = [[sg.Text("— Log —", font=("",11,"bold"))], [sg.Multiline(size=(100,20), key="LOG", autoscroll=True, disabled=True, reroute_stdout=True, reroute_stderr=True)]]

        layout = [[sg.TabGroup([[sg.Tab("List Generation", tab_list), sg.Tab("Manual Measurement", manual), sg.Tab("Protocol Builder", [[sg.Column(proto_editor), sg.VSeperator(), sg.Column(proto_view)]]), sg.Tab("Output Log", logs)]])]]
        return sg.Window("Standalone Measurement (Vipsa-based)", layout, finalize=True)

    # ---- Helpers
    def _get_float(self, key, default=None):
        try:
            s = self.window[key].get()
            return float(s) if s != "" else default
        except Exception:
            return default

    def _get_int(self, key, default=None):
        try:
            s = self.window[key].get()
            return int(float(s)) if s != "" else default
        except Exception:
            return default

    def _validate_file(self, key, label="File"):
        p = self.window[key].get()
        if not p or not os.path.exists(p):
            sg.popup_error(f"{label} not found:\n{p}")
            return None
        return p

    def _run_bg(self, fn, *args, **kwargs):
        threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()

    def _discover_visa_instruments(self):
        rm = pyvisa.ResourceManager()
        resources = list(rm.list_resources())
        discovered = {}
        unmatched = []

        for resource in resources:
            inst = None
            try:
                inst = rm.open_resource(resource)
                inst.timeout = 3000
                idn = inst.query("*IDN?").strip()
            except Exception:
                unmatched.append(resource)
                continue
            finally:
                if inst is not None:
                    try:
                        inst.close()
                    except Exception:
                        pass

            idn_upper = idn.upper()
            if "707B" in idn_upper:
                discovered["switch"] = resource
            elif "2450" in idn_upper:
                discovered["keithley"] = resource
            elif "KEYSIGHT" in idn_upper or "B290" in idn_upper:
                discovered["keysight"] = resource
            else:
                unmatched.append(resource)

        if "switch" not in discovered:
            known = {value for key, value in discovered.items() if key in {"keithley", "keysight"}}
            remaining = [resource for resource in resources if resource not in known]
            if len(remaining) == 1:
                discovered["switch"] = remaining[0]

        return discovered

    def _get_selected_smu(self, label):
        if not self.is_smu_connected:
            raise RuntimeError("SMUs are not connected.")

        key = str(label or "keysight").strip().lower()
        smu = self.smus.get(key)
        if smu is None:
            raise RuntimeError(f"Requested SMU '{label}' is not connected.")
        return smu

    def _build_smu(self, label):
        key = str(label).strip().lower()
        if key not in self.discovered_instruments:
            raise RuntimeError(f"No detected VISA address for '{label}'.")
        if self.switch is None:
            raise RuntimeError("707B switch is not connected.")

        address = self.discovered_instruments[key]
        if key == "keithley":
            return KeithleySMU(device_no=0, address=address, switch=self.switch, switch_channel="keithley")
        if key == "keysight":
            return KeysightSMU(device_no=0, address=address, switch=self.switch, switch_channel="keysight")
        raise RuntimeError(f"Unsupported SMU label '{label}'.")

    def _disconnect_smu_object(self, label):
        key = str(label).strip().lower()
        smu = self.smus.pop(key, None)
        if smu is None:
            return
        try:
            close_session = getattr(smu, "close_session", None)
            if callable(close_session):
                close_session()
        except Exception as exc:
            print(f"Warning: could not fully close {key} session: {exc}")

    def _set_manual_active_smu(self, label, force_reconnect=False):
        if not self.discovered_instruments or self.switch is None:
            raise RuntimeError("Equipment must be detected and connected first.")

        key = str(label).strip().lower()
        if key not in {"keithley", "keysight"}:
            raise RuntimeError(f"Unknown SMU selection '{label}'.")

        if self.active_smu_label == key and key in self.smus and not force_reconnect:
            self.smu = self.smus[key]
            self.vipsa.SMU = self.smu
            return self.smu

        if self.switch is not None:
            self.switch.open_all()

        if self.active_smu_label is not None and self.active_smu_label != key:
            self._disconnect_smu_object(self.active_smu_label)

        if key not in self.smus or force_reconnect:
            self._disconnect_smu_object(key)
            self.smus[key] = self._build_smu(key)

        self.active_smu_label = key
        self.smu = self.smus[key]
        self.vipsa.SMU = self.smu
        self.window["ACTIVE_SMU"].update(value=key.capitalize())
        self.window["IV_SMU"].update(value=key.capitalize())
        self.window["PULSE_SMU"].update(value=key.capitalize())
        self.window["SMU_STATUS"].update(f"Status: Active manual SMU = {key.capitalize()}", text_color="green")
        print(f"Manual active SMU switched to {key.capitalize()} at {self.discovered_instruments[key]}")
        return self.smu

    def _ensure_protocol_smus(self, labels):
        protocol_smus = {}
        for label in sorted({str(item).strip().lower() for item in labels if item}):
            if label not in self.smus:
                self.smus[label] = self._build_smu(label)
            protocol_smus[label] = self.smus[label]
        return protocol_smus

    # ---- Events
    def on_connect(self):
        if self.is_smu_connected:
            print("Equipment already connected.")
            return
        print("Connecting SMU …")
        try:
            discovered = self._discover_visa_instruments()
            missing = [name for name in ("switch", "keithley", "keysight") if name not in discovered]
            if missing:
                raise RuntimeError(f"Could not auto-detect: {', '.join(missing)}")

            self.discovered_instruments = discovered
            self.switch = Keithley707B(address=discovered["switch"], slot=1)
            self.switch.open_all()
            self.smus = {}
            self.is_smu_connected = True
            print(f"Detected 707B switch at {discovered['switch']}")
            print(f"Detected Keithley SMU at {discovered['keithley']}")
            print(f"Detected Keysight SMU at {discovered['keysight']}")
            requested_label = self.window["ACTIVE_SMU"].get()
            self._set_manual_active_smu(requested_label, force_reconnect=True)
        except Exception as e:
            self.is_smu_connected = False
            self.smu = None
            self.smus = {}
            self.switch = None
            self.discovered_instruments = {}
            self.active_smu_label = None
            self.window["SMU_STATUS"].update("Status: Connection Failed", text_color="red")
            sg.popup_error(f"SMU connection error:\n{e}")

    def on_disconnect(self):
        if not self.is_smu_connected:
            print("Equipment already disconnected.")
            return
        if self.switch is not None:
            try:
                self.switch.open_all()
            except Exception:
                pass
        for label in list(self.smus.keys()):
            self._disconnect_smu_object(label)
        self.smu = None
        self.smus = {}
        self.switch = None
        self.discovered_instruments = {}
        self.active_smu_label = None
        self.vipsa.SMU = None
        self.is_smu_connected = False
        self.window["SMU_STATUS"].update("Status: Disconnected", text_color="red")
        print("All equipment disconnected.")

    def on_switch_instrument(self):
        if not self.is_smu_connected:
            sg.popup_error("Connect all equipment first.")
            return

        try:
            self._set_manual_active_smu(self.window["ACTIVE_SMU"].get(), force_reconnect=True)
        except Exception as exc:
            sg.popup_error(f"Could not switch instrument:\n{exc}")

    def on_iv_preview(self, values):
        try:
            fwd = float(values["IV_FWD"]); rst = float(values["IV_RST"])
            step = float(values["IV_STEP"]); dly = float(values["IV_DELAY"]) 
            cyc = int(values["IV_CYC"])
            use_form = values["IV_FORM"]
            form_v = float(values["IV_FORM_V"]) if use_form and values["IV_FORM_V"] else None
            t, v = ListmakerHelpers.generate_iv_times_voltages(fwd, rst, step, dly, use_form, form_v, cyc)
            self.generated_iv = (t, v)
            ListmakerHelpers.plot_preview(t, v, "Generated IV Sweep")
        except Exception as e:
            sg.popup_error(f"IV preview failed:\n{e}")

    def on_iv_save(self):
        if not self.generated_iv:
            sg.popup_error("No IV list generated. Preview first.")
            return
        path = sg.popup_get_file("Save IV Sweep CSV…", save_as=True, no_window=True, file_types=(("CSV","*.csv"),), default_path="sweep_list.csv")
        if path:
            ok = ListmakerHelpers.save_to_csv(*self.generated_iv, path)
            if ok:
                self.window["SWEEP_PATH"].update(path)

    def on_pulse_preview(self, values):
        try:
            t, v = ListmakerHelpers.generate_pulse_times_voltages(
                int(values["PS_N"]), float(values["PS_V"]), float(values["PS_W"]),
                int(values["PR_N"]), float(values["PR_V"]), float(values["PR_W"]),
                int(values["PE_N"]), float(values["PE_V"]), float(values["PE_W"]),
                int(values["P_CYC"]))
            self.generated_pulse = (t, v)
            ListmakerHelpers.plot_preview(t, v, "Generated Pulse List")
        except Exception as e:
            sg.popup_error(f"Pulse preview failed:\n{e}")

    def on_pulse_save(self):
        if not self.generated_pulse:
            sg.popup_error("No pulse list generated. Preview first.")
            return
        path = sg.popup_get_file("Save Pulse CSV…", save_as=True, no_window=True, file_types=(("CSV","*.csv"),), default_path="pulse_list.csv")
        if path:
            ok = ListmakerHelpers.save_to_csv(*self.generated_pulse, path)
            if ok:
                self.window["PULSE_PATH"].update(path)

    # ---- Measurement via Vipsa_Methods
    def run_dciv(self, values):
        if not self.is_smu_connected or not self.smu:
            sg.popup_error("SMU not connected.")
            return
        sweep_csv = self._validate_file("SWEEP_PATH", "Sweep CSV")
        if not sweep_csv:
            return
        smu = self._set_manual_active_smu(values.get("ACTIVE_SMU"))
        # Collect params
        sample = values["SAMPLE"]; device = values["DEVICE"]; save_dir = values["SAVE_DIR"]
        pos = self._get_float("POS", 0.001); neg = self._get_float("NEG", 0.01)
        sw_delay = self._get_float("SW_DELAY", None)
        do_plot = values["PLOT_IV"]
        # Non-mechanical run
        def task():
            print(f"Starting DCIV: {sample}-{device} …")
            try:
                is_meas, z_h, saved_path = self.vipsa.run_single_DCIV(
                    sample_no=sample,
                    device_no=device,
                    pos_compl=pos,
                    neg_compl=neg,
                    sweep_delay=sw_delay,
                    acq_delay=None,
                    plot=do_plot,
                    align=False,
                    approach=False,
                    zaber_corr=False,
                    corr_recheck=False,
                    save_directory=save_dir,
                    sweep_path=sweep_csv,
                    SMU=smu,
                    stage=None, Zaber_x=None, Zaber_y=None, top_light=None,
                )
                if is_meas:
                    print(f"Done. Saved: {saved_path}")
                else:
                    print("DCIV aborted (no contact).")
            except Exception as e:
                print("DCIV error:", e)
        self._run_bg(task)

    def run_pulse(self, values):
        if not self.is_smu_connected or not self.smu:
            sg.popup_error("SMU not connected.")
            return
        pulse_csv = self._validate_file("PULSE_PATH", "Pulse CSV")
        if not pulse_csv:
            return
        smu = self._set_manual_active_smu(values.get("ACTIVE_SMU"))
        sample = values["SAMPLE"]; device = values["DEVICE"]; save_dir = values["SAVE_DIR"]
        comp = self._get_float("COMP", 0.01); pwidth = self._get_float("PWIDTH", None)
        do_plot = values["PLOT_P"]
        def task():
            print(f"Starting Pulse: {sample}-{device} …")
            try:
                is_meas, z_h, saved_path = self.vipsa.run_single_pulse(
                    sample_no=sample,
                    device_no=device,
                    compliance=comp,
                    pulse_width=pwidth,
                    plot=do_plot,
                    align=False,
                    approach=False,
                    zaber_corr=False,
                    corr_recheck=False,
                    save_directory=save_dir,
                    pulse_path=pulse_csv,
                    SMU=smu,
                    stage=None, Zaber_x=None, Zaber_y=None, top_light=None,
                )
                if is_meas:
                    print(f"Done. Saved: {saved_path}")
                else:
                    print("Pulse aborted (no contact).")
            except Exception as e:
                print("Pulse error:", e)
        self._run_bg(task)

    # ---- Protocol (each step delegates to Vipsa methods)
    def add_step(self, values):
        t = values["P_TYPE"]
        if not t:
            sg.popup_error("Select a test type first.")
            return
        if t == "DCIV from File":
            layout = [[sg.Text("Sweep CSV"), sg.Input(key="PATH"), sg.FileBrowse()],
                      [sg.Text("SMU"), sg.Combo(["Keithley", "Keysight"], default_value="Keithley", key="SMU", readonly=True)],
                      [sg.Text("Pos Comp"), sg.Input("0.001", key="POS"), sg.Text("Neg Comp"), sg.Input("0.01", key="NEG")],
                      [sg.Text("Delay (opt)"), sg.Input("", key="DELAY")], [sg.Submit(), sg.Cancel()]]
        else:
            layout = [[sg.Text("Pulse CSV"), sg.Input(key="PATH"), sg.FileBrowse()],
                      [sg.Text("SMU"), sg.Combo(["Keysight", "Keithley"], default_value="Keysight", key="SMU", readonly=True)],
                      [sg.Text("Compliance"), sg.Input("0.01", key="COMP")],
                      [sg.Text("Width (opt)"), sg.Input("", key="WIDTH")], [sg.Submit(), sg.Cancel()]]
        w = sg.Window("Configure Step", layout)
        ev, vals = w.read(); w.close()
        if ev != "Submit":
            return
        path = vals.get("PATH", "")
        if not path or not os.path.exists(path):
            sg.popup_error("Invalid path.")
            return
        step = {"type": t, "path": path, "smu": vals.get("SMU", "Keysight").strip().lower()}
        if t == "DCIV from File":
            step.update({"pos": float(vals["POS"]), "neg": float(vals["NEG"]), "delay": float(vals["DELAY"]) if vals["DELAY"] else None})
        else:
            step.update({"comp": float(vals["COMP"]), "width": float(vals["WIDTH"]) if vals["WIDTH"] else None})
        cur = list(self.window["P_LIST"].get_list_values())
        cur.append(step)
        self.window["P_LIST"].update(cur)

    def remove_step(self):
        cur = list(self.window["P_LIST"].get_list_values())
        sel = self.window["P_LIST"].get()
        if not sel:
            return
        cur.remove(sel[0])
        self.window["P_LIST"].update(cur)

    def clear_steps(self):
        self.window["P_LIST"].update([])

    def run_protocol(self, values):
        if not self.is_smu_connected or not self.smu:
            sg.popup_error("SMU not connected.")
            return
        steps = list(self.window["P_LIST"].get_list_values())
        if not steps:
            sg.popup_error("No steps to run.")
            return
        sample = values["P_SAMPLE"]; device = values["P_DEVICE"]; save_dir = values["P_SAVE"]
        do_plot = values["P_PLOT"]
        def task():
            try:
                protocol = []
                for st in steps:
                    if st["type"] == "DCIV from File":
                        protocol.append({
                            "type": "DCIV",
                            "params": {
                                "smu": st.get("smu", "keithley"),
                                "sweep_path": st["path"],
                                "pos_compl": st["pos"],
                                "neg_compl": st["neg"],
                                "sweep_delay": st["delay"],
                                "align": False,
                                "approach": False,
                            },
                        })
                    else:
                        protocol.append({
                            "type": "PULSE",
                            "params": {
                                "smu": st.get("smu", "keysight"),
                                "pulse_path": st["path"],
                                "compliance": st["comp"],
                                "pulse_width": st["width"],
                                "align": False,
                                "approach": False,
                            },
                        })

                protocol_smus = self._ensure_protocol_smus(
                    step["params"].get("smu") for step in protocol
                )
                results = self.vipsa.run_protocol(
                    protocol,
                    sample_no=sample,
                    device_no=device,
                    save_directory=save_dir,
                    SMU=protocol_smus,
                    stage=None,
                    Zaber_x=None,
                    Zaber_y=None,
                    top_light=None,
                )
                if do_plot:
                    print("Protocol plotting is not yet auto-expanded per mixed-SMU step; saved files are still written by Main4.")
                print("Protocol results:", results)
                print("Protocol complete.")
            except Exception as e:
                print("Protocol error:", e)
        self._run_bg(task)

    # ---- Main loop
    def run(self):
        while True:
            ev, vals = self.window.read()
            if ev in (sg.WIN_CLOSED, None):
                if self.is_smu_connected:
                    self.on_disconnect()
                break
            if ev == "Connect All Equipment":
                self.on_connect()
            elif ev == "Disconnect All Equipment":
                self.on_disconnect()
            elif ev == "Switch Instrument":
                self.on_switch_instrument()
            elif ev == "IV_FORM":
                self.window["IV_FORM_V"].update(disabled=not vals["IV_FORM"])
            elif ev == "Preview IV":
                self.on_iv_preview(vals)
            elif ev == "Save IV CSV":
                self.on_iv_save()
            elif ev == "Preview Pulse":
                self.on_pulse_preview(vals)
            elif ev == "Save Pulse CSV":
                self.on_pulse_save()
            elif ev == "Run DCIV":
                self.run_dciv(vals)
            elif ev == "Run Pulse":
                self.run_pulse(vals)
            elif ev == "Add Step":
                self.add_step(vals)
            elif ev == "Remove Step":
                self.remove_step()
            elif ev == "Clear":
                self.clear_steps()
            elif ev == "Run Protocol":
                self.run_protocol(vals)
        self.window.close()


if __name__ == "__main__":
    app = StandaloneTesterGUI()
    app.run()
