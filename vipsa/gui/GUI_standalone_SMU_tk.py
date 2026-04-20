from __future__ import annotations

import csv
import json
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib.pyplot as plt
import pyvisa
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from vipsa.analysis.Datahandling import Data_Handler
from vipsa.gui.ProtocolEditor_tk import ProtocolBuilderTk
from vipsa.hardware.Source_Measure_Unit import Keithley707B, KeithleySMU, KeysightSMU
from vipsa.workflows.Main4 import Vipsa_Methods


DEFAULT_SAVE_DIRECTORY = "./manual_probe_data"
DEFAULT_SWEEP_PATH = ""
DEFAULT_PULSE_PATH = ""


class ListmakerHelpers:
    @staticmethod
    def generate_iv_times_voltages(
        forward_v: float,
        reset_v: float,
        step_v: float,
        step_delay: float,
        forming: bool,
        forming_v: float | None,
        cycles: int,
    ):
        times = []
        voltages = []
        current_time = 0.0

        def up_down(vmax):
            nonlocal current_time
            vv = 0.0
            while vv <= vmax:
                voltages.append(vv)
                times.append(current_time)
                current_time += step_delay
                vv += step_v
            vv -= step_v
            while vv >= 0:
                voltages.append(vv)
                times.append(current_time)
                current_time += step_delay
                vv -= step_v

        def down_up(vmin):
            nonlocal current_time
            vv = 0.0
            while vv >= vmin:
                voltages.append(vv)
                times.append(current_time)
                current_time += step_delay
                vv -= step_v
            vv += step_v
            while vv <= 0:
                voltages.append(vv)
                times.append(current_time)
                current_time += step_delay
                vv += step_v

        if forming and forming_v is not None:
            up_down(forming_v)
            down_up(reset_v)

        count = cycles if not forming else max(cycles - 1, 0)
        for _ in range(count):
            up_down(forward_v)
            down_up(reset_v)

        if voltages and voltages[-1] != 0:
            voltages.append(0.0)
            times.append(current_time)
        return times, voltages

    @staticmethod
    def generate_pulse_times_voltages(
        set_n,
        set_v,
        set_w,
        read_n,
        read_v,
        read_w,
        erase_n,
        erase_v,
        erase_w,
        cycles,
    ):
        times = []
        voltages = []
        current_time = 0.0
        for _ in range(int(cycles)):
            for _ in range(int(set_n)):
                voltages.append(set_v)
                times.append(current_time)
                current_time += set_w
                voltages.append(0.0)
                times.append(current_time)
            for _ in range(int(read_n)):
                voltages.append(read_v)
                times.append(current_time)
                current_time += read_w
                voltages.append(0.0)
                times.append(current_time)
            for _ in range(int(erase_n)):
                voltages.append(erase_v)
                times.append(current_time)
                current_time += erase_w
                voltages.append(0.0)
                times.append(current_time)
            for _ in range(int(read_n)):
                voltages.append(read_v)
                times.append(current_time)
                current_time += read_w
                voltages.append(0.0)
                times.append(current_time)
        return times, voltages

    @staticmethod
    def save_to_csv(times, volts, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Time (s)", "Voltage (V)"])
            for time_value, voltage_value in zip(times, volts):
                writer.writerow([f"{time_value:.6f}", f"{voltage_value:.6f}"])


class StandaloneMeasurementTkApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ViPSA Standalone Measurement")
        self.root.geometry("1500x920")
        self.root.minsize(1260, 820)

        self.vipsa = Vipsa_Methods()
        self.data_handler = Data_Handler()
        self.protocol_builder = ProtocolBuilderTk(self.root, self.vipsa, logger=self._log)

        self.vars = {}
        self.widgets = {}
        self.log_queue = queue.Queue()
        self.worker_thread = None

        self.smu = None
        self.smus = {}
        self.switch = None
        self.discovered_instruments = {}
        self.active_smu_label = None
        self.is_smu_connected = False

        self.generated_iv = None
        self.generated_pulse = None
        self.live_plot_title = "Live Sweep"
        self.live_plot_points = []

        self._build_ui()
        self.root.after(120, self._flush_log_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _string_var(self, key, default=""):
        var = self.vars.get(key)
        if isinstance(var, tk.StringVar):
            return var
        var = tk.StringVar(value=default)
        self.vars[key] = var
        return var

    def _bool_var(self, key, default=False):
        var = self.vars.get(key)
        if isinstance(var, tk.BooleanVar):
            return var
        var = tk.BooleanVar(value=default)
        self.vars[key] = var
        return var

    def _build_ui(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, padding=10)
        right = ttk.Frame(main, padding=10)
        main.add(left, weight=4)
        main.add(right, weight=2)

        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill="both", expand=True)

        self._create_list_tab()
        self._create_measurement_tab()
        self._create_protocol_tab()
        self._create_log_tab()

        self._create_live_plot_panel(right)
        self._create_plot_actions(right)

    def _create_list_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="List Generation")

        iv_box = ttk.LabelFrame(frame, text="I-V Sweep List", padding=10)
        iv_box.pack(fill="x", pady=(0, 10))
        self._add_entry(iv_box, "Forward V", "IV_FWD", "1.5", 0, 0)
        self._add_entry(iv_box, "Reset V", "IV_RST", "-1.5", 0, 2)
        self._add_entry(iv_box, "Step V", "IV_STEP", "0.05", 1, 0)
        self._add_entry(iv_box, "Delay (s)", "IV_DELAY", "0.001", 1, 2)
        self._add_entry(iv_box, "Cycles", "IV_CYC", "2", 2, 0)
        ttk.Checkbutton(
            iv_box,
            text="Forming",
            variable=self._bool_var("IV_FORM", False),
            command=self._toggle_forming_entry,
        ).grid(row=2, column=2, sticky="w", pady=(6, 0))
        self._add_entry(iv_box, "Forming V", "IV_FORM_V", "", 2, 3)
        ttk.Button(iv_box, text="Preview IV", command=self.on_iv_preview).grid(row=3, column=0, pady=(8, 0), sticky="w")
        ttk.Button(iv_box, text="Save IV CSV", command=self.on_iv_save).grid(row=3, column=1, pady=(8, 0), sticky="w")

        pulse_box = ttk.LabelFrame(frame, text="Pulse List", padding=10)
        pulse_box.pack(fill="x")
        self._add_entry(pulse_box, "Set N", "PS_N", "1", 0, 0)
        self._add_entry(pulse_box, "Set V", "PS_V", "1.5", 0, 2)
        self._add_entry(pulse_box, "Set W", "PS_W", "0.001", 0, 4)
        self._add_entry(pulse_box, "Read N", "PR_N", "1", 1, 0)
        self._add_entry(pulse_box, "Read V", "PR_V", "0.1", 1, 2)
        self._add_entry(pulse_box, "Read W", "PR_W", "0.001", 1, 4)
        self._add_entry(pulse_box, "Erase N", "PE_N", "1", 2, 0)
        self._add_entry(pulse_box, "Erase V", "PE_V", "-1.5", 2, 2)
        self._add_entry(pulse_box, "Erase W", "PE_W", "0.001", 2, 4)
        self._add_entry(pulse_box, "Cycles", "P_CYC", "10", 3, 0)
        ttk.Button(pulse_box, text="Preview Pulse", command=self.on_pulse_preview).grid(row=4, column=0, pady=(8, 0), sticky="w")
        ttk.Button(pulse_box, text="Save Pulse CSV", command=self.on_pulse_save).grid(row=4, column=1, pady=(8, 0), sticky="w")
        self._toggle_forming_entry()

    def _create_measurement_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Manual Measurement")

        conn = ttk.LabelFrame(frame, text="Connections", padding=10)
        conn.pack(fill="x", pady=(0, 10))
        ttk.Button(conn, text="Connect All Equipment", command=self.on_connect).grid(row=0, column=0, sticky="w")
        ttk.Button(conn, text="Disconnect All Equipment", command=self.on_disconnect).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(conn, textvariable=self._string_var("SMU_STATUS", "Status: Disconnected")).grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Label(conn, text="Active SMU").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            conn,
            textvariable=self._string_var("ACTIVE_SMU", "Keithley"),
            values=("Keithley", "Keysight"),
            state="readonly",
            width=12,
        ).grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Button(conn, text="Switch Instrument", command=self.on_switch_instrument).grid(row=1, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        meta = ttk.LabelFrame(frame, text="Run Context", padding=10)
        meta.pack(fill="x", pady=(0, 10))
        self._add_entry(meta, "Sample", "SAMPLE", "SampleX", 0, 0)
        self._add_entry(meta, "Device", "DEVICE", "Dev1", 0, 2)
        self._add_entry(meta, "Operator", "OPERATOR", "", 1, 0)
        self._add_entry(meta, "Batch", "BATCH", "", 1, 2)
        ttk.Label(meta, text="Save Folder").grid(row=2, column=0, sticky="w", pady=(6, 0))
        folder_row = ttk.Frame(meta)
        folder_row.grid(row=2, column=1, columnspan=3, sticky="ew", pady=(6, 0))
        folder_row.columnconfigure(0, weight=1)
        ttk.Entry(folder_row, textvariable=self._string_var("SAVE_DIR", DEFAULT_SAVE_DIRECTORY)).grid(row=0, column=0, sticky="ew")
        ttk.Button(folder_row, text="Browse", command=lambda: self._choose_directory("SAVE_DIR")).grid(row=0, column=1, padx=(6, 0))
        ttk.Label(meta, text="Notes").grid(row=3, column=0, sticky="nw", pady=(6, 0))
        self.notes_text = tk.Text(meta, height=4, wrap="word")
        self.notes_text.grid(row=3, column=1, columnspan=3, sticky="ew", pady=(6, 0))
        for col in range(4):
            meta.columnconfigure(col, weight=1 if col % 2 else 0)

        dciv = ttk.LabelFrame(frame, text="Run DCIV from CSV", padding=10)
        dciv.pack(fill="x", pady=(0, 10))
        ttk.Label(dciv, text="Sweep CSV").grid(row=0, column=0, sticky="w")
        sweep_row = ttk.Frame(dciv)
        sweep_row.grid(row=0, column=1, columnspan=5, sticky="ew")
        sweep_row.columnconfigure(0, weight=1)
        ttk.Entry(sweep_row, textvariable=self._string_var("SWEEP_PATH", DEFAULT_SWEEP_PATH)).grid(row=0, column=0, sticky="ew")
        ttk.Button(sweep_row, text="Browse", command=lambda: self._choose_file("SWEEP_PATH", [("CSV Files", "*.csv")])).grid(row=0, column=1, padx=(6, 0))
        self._add_entry(dciv, "Pos Comp (A)", "POS", "0.001", 1, 0)
        self._add_entry(dciv, "Neg Comp (A)", "NEG", "0.01", 1, 2)
        self._add_entry(dciv, "Sweep Delay", "SW_DELAY", "0.0001", 1, 4)
        ttk.Checkbutton(dciv, text="Use Current Autorange", variable=self._bool_var("DCIV_AUTORANGE", False)).grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(dciv, text="Use 4-way Split", variable=self._bool_var("DCIV_USE_4WAY", True)).grid(row=2, column=1, sticky="w", pady=(6, 0))
        ttk.Checkbutton(dciv, text="Include Read Probe", variable=self._bool_var("DCIV_READ_PROBE", True)).grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Button(dciv, text="Run DCIV", command=self.queue_run_dciv).grid(row=3, column=0, sticky="w", pady=(8, 0))

        pulse = ttk.LabelFrame(frame, text="Run Pulse from CSV", padding=10)
        pulse.pack(fill="x")
        ttk.Label(pulse, text="Pulse CSV").grid(row=0, column=0, sticky="w")
        pulse_row = ttk.Frame(pulse)
        pulse_row.grid(row=0, column=1, columnspan=5, sticky="ew")
        pulse_row.columnconfigure(0, weight=1)
        ttk.Entry(pulse_row, textvariable=self._string_var("PULSE_PATH", DEFAULT_PULSE_PATH)).grid(row=0, column=0, sticky="ew")
        ttk.Button(pulse_row, text="Browse", command=lambda: self._choose_file("PULSE_PATH", [("CSV Files", "*.csv")])).grid(row=0, column=1, padx=(6, 0))
        self._add_entry(pulse, "Compliance (A)", "COMP", "0.01", 1, 0)
        self._add_entry(pulse, "Pulse Width", "PWIDTH", "", 1, 2)
        self._add_entry(pulse, "Acquire Delay", "P_ACQ", "", 1, 4)
        ttk.Checkbutton(pulse, text="Use Current Autorange", variable=self._bool_var("PULSE_AUTORANGE", False)).grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Button(pulse, text="Run Pulse", command=self.queue_run_pulse).grid(row=3, column=0, sticky="w", pady=(8, 0))

        probe = ttk.LabelFrame(frame, text="Constant Voltage Current Probe", padding=10)
        probe.pack(fill="x", pady=(10, 0))
        self._add_entry(probe, "Voltage (V)", "CP_VOLTAGE", "0.1", 0, 0)
        self._add_entry(probe, "Duration (s)", "CP_DURATION", "1.0", 0, 2)
        self._add_entry(probe, "Sample Interval (s)", "CP_INTERVAL", "0.1", 1, 0)
        self._add_entry(probe, "Compliance (A)", "CP_COMP", "0.001", 1, 2)
        ttk.Checkbutton(probe, text="Use Current Autorange", variable=self._bool_var("CP_AUTORANGE", False)).grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Button(probe, text="Run Current Probe", command=self.queue_run_current_probe).grid(row=3, column=0, sticky="w", pady=(8, 0))

    def _create_protocol_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Protocol")
        left = ttk.Frame(frame)
        left.pack(side="left", fill="y")
        right = ttk.Frame(frame)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        ttk.Button(left, text="Add Step", command=self._add_protocol_step).pack(fill="x", pady=(0, 4))
        ttk.Button(left, text="Edit Selected", command=self._edit_protocol_step).pack(fill="x", pady=4)
        ttk.Button(left, text="Remove Selected", command=self._remove_protocol_step).pack(fill="x", pady=4)
        ttk.Button(left, text="Clear", command=self._clear_protocol).pack(fill="x", pady=4)
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        ttk.Button(left, text="Save Protocol", command=self._save_protocol).pack(fill="x", pady=4)
        ttk.Button(left, text="Load Protocol", command=self._load_protocol).pack(fill="x", pady=4)
        ttk.Button(left, text="Run Protocol", command=self.queue_run_protocol).pack(fill="x", pady=(12, 0))

        list_frame = ttk.Frame(right)
        list_frame.pack(fill="both", expand=True)
        self.protocol_listbox = tk.Listbox(list_frame, exportselection=False)
        self.protocol_listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.protocol_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.protocol_listbox.configure(yscrollcommand=scrollbar.set)

    def _create_log_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Output Log")
        self.log_text = scrolledtext.ScrolledText(frame, wrap="word", height=30)
        self.log_text.pack(fill="both", expand=True)

    def _create_live_plot_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Live Plot", padding=10)
        panel.pack(fill="both", expand=True)
        self.live_plot_figure = Figure(figsize=(4.4, 3.2), dpi=100)
        self.live_plot_axis = self.live_plot_figure.add_subplot(111)
        self._reset_live_plot("Live Sweep")
        self.live_plot_canvas = FigureCanvasTkAgg(self.live_plot_figure, master=panel)
        self.live_plot_canvas.draw_idle()
        self.live_plot_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _create_plot_actions(self, parent):
        panel = ttk.LabelFrame(parent, text="Graph Actions", padding=10)
        panel.pack(fill="x", pady=(10, 0))
        ttk.Button(panel, text="Save Current Graph", command=self._save_current_graph).pack(fill="x")
        ttk.Button(panel, text="Reset Plot", command=lambda: self._reset_live_plot("Live Sweep")).pack(fill="x", pady=(6, 0))

    def _add_entry(self, parent, label, key, default, row, column):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", pady=(6, 0) if row else 0)
        entry = ttk.Entry(parent, textvariable=self._string_var(key, default), width=16)
        entry.grid(
            row=row, column=column + 1, sticky="ew", padx=(6, 10), pady=(6, 0) if row else 0
        )
        self.widgets[key] = entry

    def _choose_file(self, key, filetypes):
        path = filedialog.askopenfilename(parent=self.root, filetypes=filetypes)
        if path:
            self._string_var(key).set(path)

    def _choose_directory(self, key):
        path = filedialog.askdirectory(parent=self.root)
        if path:
            self._string_var(key).set(path)

    def _toggle_forming_entry(self):
        state = "normal" if self._bool_var("IV_FORM").get() else "disabled"
        entry = self.widgets.get("IV_FORM_V")
        if entry is not None:
            entry.configure(state=state)

    def _discover_visa_instruments(self):
        rm = pyvisa.ResourceManager()
        resources = list(rm.list_resources())
        discovered = {}
        for resource in resources:
            inst = None
            try:
                inst = rm.open_resource(resource)
                inst.timeout = 3000
                idn = inst.query("*IDN?").strip().upper()
            except Exception:
                continue
            finally:
                if inst is not None:
                    try:
                        inst.close()
                    except Exception:
                        pass
            if "707B" in idn:
                discovered["switch"] = resource
            elif "2450" in idn:
                discovered["keithley"] = resource
            elif "KEYSIGHT" in idn or "B290" in idn:
                discovered["keysight"] = resource
        return discovered

    def _build_smu(self, label):
        key = str(label).strip().lower()
        address = self.discovered_instruments[key]
        if key == "keithley":
            return KeithleySMU(device_no=0, address=address, switch=self.switch, switch_channel="keithley")
        if key == "keysight":
            return KeysightSMU(device_no=0, address=address, switch=self.switch, switch_channel="keysight")
        raise RuntimeError(f"Unsupported SMU selection '{label}'.")

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
            self._log(f"Warning while closing {label}: {exc}\n")

    def _set_manual_active_smu(self, label, force_reconnect=False):
        if not self.is_smu_connected:
            raise RuntimeError("Connect equipment first.")
        key = str(label or "keithley").strip().lower()
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
        self._string_var("SMU_STATUS").set(f"Status: Active SMU = {key.capitalize()}")
        return self.smu

    def _ensure_protocol_smus(self, labels):
        protocol_smus = {}
        for label in sorted({str(item).strip().lower() for item in labels if item}):
            if label not in self.smus:
                self.smus[label] = self._build_smu(label)
            protocol_smus[label] = self.smus[label]
        return protocol_smus

    def on_connect(self):
        try:
            self.discovered_instruments = self._discover_visa_instruments()
            missing = [name for name in ("switch", "keithley", "keysight") if name not in self.discovered_instruments]
            if missing:
                raise RuntimeError(f"Could not auto-detect: {', '.join(missing)}")
            self.switch = Keithley707B(address=self.discovered_instruments["switch"], slot=1)
            self.switch.open_all()
            self.smus = {}
            self.is_smu_connected = True
            self._set_manual_active_smu(self._string_var("ACTIVE_SMU").get(), force_reconnect=True)
            self._log("Connected switch and both SMUs.\n")
        except Exception as exc:
            self.is_smu_connected = False
            self.smu = None
            self.smus = {}
            self.switch = None
            self.discovered_instruments = {}
            self.active_smu_label = None
            self._string_var("SMU_STATUS").set("Status: Connection Failed")
            messagebox.showerror("Connection Failed", str(exc), parent=self.root)

    def on_disconnect(self):
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
        self._string_var("SMU_STATUS").set("Status: Disconnected")
        self._log("Disconnected all equipment.\n")

    def on_switch_instrument(self):
        try:
            self._set_manual_active_smu(self._string_var("ACTIVE_SMU").get(), force_reconnect=True)
            self._log(f"Switched active SMU to {self._string_var('ACTIVE_SMU').get()}.\n")
        except Exception as exc:
            messagebox.showerror("Switch Failed", str(exc), parent=self.root)

    def on_iv_preview(self):
        try:
            times, voltages = ListmakerHelpers.generate_iv_times_voltages(
                float(self._string_var("IV_FWD").get()),
                float(self._string_var("IV_RST").get()),
                float(self._string_var("IV_STEP").get()),
                float(self._string_var("IV_DELAY").get()),
                bool(self._bool_var("IV_FORM").get()),
                float(self._string_var("IV_FORM_V").get()) if self._bool_var("IV_FORM").get() and self._string_var("IV_FORM_V").get() else None,
                int(float(self._string_var("IV_CYC").get())),
            )
            self.generated_iv = (times, voltages)
            plt.figure(figsize=(10, 4))
            plt.plot(times, voltages)
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.title("Generated IV Sweep")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            messagebox.showerror("Preview Failed", str(exc), parent=self.root)

    def on_iv_save(self):
        if not self.generated_iv:
            messagebox.showerror("Missing Preview", "Preview an IV list first.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".csv", initialfile="sweep_list.csv", filetypes=(("CSV Files", "*.csv"),))
        if path:
            ListmakerHelpers.save_to_csv(*self.generated_iv, path)
            self._string_var("SWEEP_PATH").set(path)

    def on_pulse_preview(self):
        try:
            times, voltages = ListmakerHelpers.generate_pulse_times_voltages(
                int(float(self._string_var("PS_N").get())),
                float(self._string_var("PS_V").get()),
                float(self._string_var("PS_W").get()),
                int(float(self._string_var("PR_N").get())),
                float(self._string_var("PR_V").get()),
                float(self._string_var("PR_W").get()),
                int(float(self._string_var("PE_N").get())),
                float(self._string_var("PE_V").get()),
                float(self._string_var("PE_W").get()),
                int(float(self._string_var("P_CYC").get())),
            )
            self.generated_pulse = (times, voltages)
            plt.figure(figsize=(10, 4))
            plt.plot(times, voltages)
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.title("Generated Pulse List")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            messagebox.showerror("Preview Failed", str(exc), parent=self.root)

    def on_pulse_save(self):
        if not self.generated_pulse:
            messagebox.showerror("Missing Preview", "Preview a pulse list first.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".csv", initialfile="pulse_list.csv", filetypes=(("CSV Files", "*.csv"),))
        if path:
            ListmakerHelpers.save_to_csv(*self.generated_pulse, path)
            self._string_var("PULSE_PATH").set(path)

    def _measurement_context(self):
        return {
            "operator": self._string_var("OPERATOR").get().strip(),
            "batch": self._string_var("BATCH").get().strip(),
            "notes": self.notes_text.get("1.0", tk.END).strip(),
            "active_smu_label": self._string_var("ACTIVE_SMU").get().strip(),
        }

    def _augment_saved_metadata(self, saved_path, extra_metadata):
        if not saved_path:
            return
        metadata_path = f"{os.path.splitext(saved_path)[0]}.metadata.json"
        payload = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                payload = {}
        payload.setdefault("gui_context", {}).update(extra_metadata)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def _flush_log_queue(self):
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert(tk.END, message)
            self.log_text.see(tk.END)
        self.root.after(120, self._flush_log_queue)

    def _log(self, message):
        self.log_queue.put(str(message))

    def _run_in_background(self, label, target):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showerror("Busy", "A background task is already running.", parent=self.root)
            return

        def runner():
            self._log(f"{label} started.\n")
            try:
                target()
            except Exception as exc:
                self._log(f"{label} failed: {exc}\n")
            else:
                self._log(f"{label} finished.\n")

        self.worker_thread = threading.Thread(target=runner, daemon=True)
        self.worker_thread.start()

    def _reset_live_plot(self, title, x_label="Voltage (V)"):
        self.live_plot_title = title
        self.live_plot_x_label = x_label
        self.live_plot_points = []
        self.live_plot_axis.clear()
        self.live_plot_axis.set_title(title)
        self.live_plot_axis.set_xlabel(x_label)
        self.live_plot_axis.set_ylabel("|Current| (A)")
        self.live_plot_axis.set_yscale("log")
        self.live_plot_axis.grid(True, alpha=0.4)
        (self.live_plot_line,) = self.live_plot_axis.plot([], [], color="#198754", linewidth=1.4)
        self.live_plot_canvas.draw_idle()

    def _apply_live_plot_chunk(self, chunk, label=None):
        items = list(chunk or [])
        x_label = None
        if items and isinstance(items[0], dict):
            x_label = items[0].get("plot_x_label")
        if label and label != self.live_plot_title:
            self._reset_live_plot(label, x_label=x_label or "Voltage (V)")
        elif x_label and x_label != getattr(self, "live_plot_x_label", "Voltage (V)"):
            self._reset_live_plot(label or self.live_plot_title, x_label=x_label)
        for item in items:
            if isinstance(item, dict):
                voltage = float(item.get("plot_x", item.get("voltage", item.get("V_cmd (V)", item.get("Voltage (V)", 0.0)))))
                current = abs(float(item.get("current", item.get("Current (A)", 0.0))))
            else:
                voltage = float(item[1])
                current = abs(float(item[2]))
            self.live_plot_points.append((voltage, max(current, 1e-15)))
        if not self.live_plot_points:
            return
        xs = [point[0] for point in self.live_plot_points]
        ys = [point[1] for point in self.live_plot_points]
        self.live_plot_line.set_data(xs, ys)
        xmin, xmax = min(xs), max(xs)
        if xmin == xmax:
            xmin -= 0.1
            xmax += 0.1
        ymin, ymax = min(ys), max(ys)
        self.live_plot_axis.set_xlim(xmin, xmax)
        self.live_plot_axis.set_ylim(max(ymin * 0.8, 1e-15), max(ymax * 1.2, 1e-14))
        self.live_plot_canvas.draw_idle()

    def _live_plot_callback(self, chunk, label=None):
        self.root.after(0, lambda: self._apply_live_plot_chunk(chunk, label))

    def _plot_saved_csv(self, csv_path, data_name):
        try:
            figure = self.data_handler.build_measurement_figure(
                csv_path,
                data_name=data_name,
                sample_id=self._string_var("SAMPLE").get(),
                device_id=self._string_var("DEVICE").get(),
            )
        except Exception as exc:
            self._log(f"Could not rebuild saved plot: {exc}\n")
            return
        self.live_plot_figure.clf()
        source_axis = figure.axes[0]
        target_axis = self.live_plot_figure.add_subplot(111)
        for line in source_axis.get_lines():
            target_axis.plot(
                line.get_xdata(),
                line.get_ydata(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                label=line.get_label(),
            )
        target_axis.set_title(source_axis.get_title())
        target_axis.set_xlabel(source_axis.get_xlabel())
        target_axis.set_ylabel(source_axis.get_ylabel())
        target_axis.set_xscale(source_axis.get_xscale())
        target_axis.set_yscale(source_axis.get_yscale())
        target_axis.grid(True, alpha=0.4)
        legend = source_axis.get_legend()
        if legend is not None:
            target_axis.legend()
        self.live_plot_axis = target_axis
        self.live_plot_canvas.draw_idle()
        plt.close(figure)

    def _save_current_graph(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".png", initialfile="measurement_plot.png", filetypes=(("PNG Files", "*.png"), ("PDF Files", "*.pdf")))
        if path:
            self.live_plot_figure.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"Saved graph to {path}\n")

    def _require_connection(self):
        if not self.is_smu_connected:
            messagebox.showerror("Not Connected", "Connect equipment first.", parent=self.root)
            return False
        return True

    def queue_run_dciv(self):
        if not self._require_connection():
            return
        self._run_in_background("Single DCIV", self._run_dciv)

    def _run_dciv(self):
        sweep_path = self._string_var("SWEEP_PATH").get()
        if not sweep_path or not os.path.exists(sweep_path):
            raise FileNotFoundError(f"Sweep CSV not found: {sweep_path}")
        smu = self._set_manual_active_smu(self._string_var("ACTIVE_SMU").get())
        context = self._measurement_context()
        self.root.after(0, lambda: self._reset_live_plot("Live DCIV"))
        measured, height, saved_path = self.vipsa.run_single_DCIV(
            sample_no=self._string_var("SAMPLE").get(),
            device_no=self._string_var("DEVICE").get(),
            pos_compl=float(self._string_var("POS").get()),
            neg_compl=float(self._string_var("NEG").get()),
            sweep_delay=float(self._string_var("SW_DELAY").get()) if self._string_var("SW_DELAY").get() else None,
            plot=False,
            align=False,
            approach=False,
            zaber_corr=False,
            corr_recheck=False,
            save_directory=self._string_var("SAVE_DIR").get(),
            sweep_path=sweep_path,
            SMU=smu,
            stage=None,
            Zaber_x=None,
            Zaber_y=None,
            top_light=None,
            current_autorange=self._bool_var("DCIV_AUTORANGE").get(),
            use_4way_split=self._bool_var("DCIV_USE_4WAY").get(),
            include_read_probe=self._bool_var("DCIV_READ_PROBE").get(),
            progress_callback=self._live_plot_callback,
        )
        if measured:
            self._augment_saved_metadata(saved_path, context)
            self._log(f"DCIV saved to {saved_path} | height={height}\n")
        else:
            self._log("DCIV did not complete successfully.\n")

    def queue_run_pulse(self):
        if not self._require_connection():
            return
        self._run_in_background("Single Pulse", self._run_pulse)

    def _run_pulse(self):
        pulse_path = self._string_var("PULSE_PATH").get()
        if not pulse_path or not os.path.exists(pulse_path):
            raise FileNotFoundError(f"Pulse CSV not found: {pulse_path}")
        smu = self._set_manual_active_smu(self._string_var("ACTIVE_SMU").get())
        context = self._measurement_context()
        self.root.after(0, lambda: self._reset_live_plot("Pulse Result"))
        measured, height, saved_path = self.vipsa.run_single_pulse(
            sample_no=self._string_var("SAMPLE").get(),
            device_no=self._string_var("DEVICE").get(),
            compliance=float(self._string_var("COMP").get()),
            pulse_width=float(self._string_var("PWIDTH").get()) if self._string_var("PWIDTH").get() else None,
            plot=False,
            align=False,
            approach=False,
            zaber_corr=False,
            corr_recheck=False,
            save_directory=self._string_var("SAVE_DIR").get(),
            pulse_path=pulse_path,
            set_acquire_delay=float(self._string_var("P_ACQ").get()) if self._string_var("P_ACQ").get() else None,
            SMU=smu,
            stage=None,
            Zaber_x=None,
            Zaber_y=None,
            top_light=None,
            current_autorange=self._bool_var("PULSE_AUTORANGE").get(),
        )
        if measured:
            self._augment_saved_metadata(saved_path, context)
            self.root.after(0, lambda: self._plot_saved_csv(saved_path, "Pulse"))
            self._log(f"Pulse saved to {saved_path} | height={height}\n")
        else:
            self._log("Pulse did not complete successfully.\n")

    def queue_run_current_probe(self):
        if not self._require_connection():
            return
        self._run_in_background("Current Probe", self._run_current_probe)

    def _run_current_probe(self):
        smu = self._set_manual_active_smu(self._string_var("ACTIVE_SMU").get())
        context = self._measurement_context()
        self.root.after(0, lambda: self._reset_live_plot("Current Probe", x_label="Time (s)"))
        measured, height, saved_path = self.vipsa.run_constant_voltage_current_probe(
            sample_no=self._string_var("SAMPLE").get(),
            device_no=self._string_var("DEVICE").get(),
            voltage=float(self._string_var("CP_VOLTAGE").get()),
            duration=float(self._string_var("CP_DURATION").get()),
            compliance=float(self._string_var("CP_COMP").get()),
            sample_interval=float(self._string_var("CP_INTERVAL").get()),
            plot=False,
            align=False,
            approach=False,
            zaber_corr=False,
            corr_recheck=False,
            save_directory=self._string_var("SAVE_DIR").get(),
            SMU=smu,
            stage=None,
            Zaber_x=None,
            Zaber_y=None,
            top_light=None,
            current_autorange=self._bool_var("CP_AUTORANGE").get(),
            progress_callback=self._live_plot_callback,
        )
        if measured:
            self._augment_saved_metadata(saved_path, context)
            self.root.after(0, lambda: self._plot_saved_csv(saved_path, "CurrentProbe"))
            self._log(f"Current probe saved to {saved_path} | height={height}\n")
        else:
            self._log("Current probe did not complete successfully.\n")

    def _selected_protocol_index(self):
        selection = self.protocol_listbox.curselection()
        return selection[0] if selection else None

    def _refresh_protocol_display(self):
        self.protocol_listbox.delete(0, tk.END)
        for item in self.protocol_builder.get_protocol_display_list():
            self.protocol_listbox.insert(tk.END, item)

    def _add_protocol_step(self):
        result = self.protocol_builder.show_step_editor()
        if result:
            self._refresh_protocol_display()
            self._log(f"{result}\n")

    def _edit_protocol_step(self):
        index = self._selected_protocol_index()
        if index is None:
            return
        result = self.protocol_builder.show_step_editor(
            initial_step=self.protocol_builder.protocol_list_configs[index],
            edit_index=index,
        )
        if result:
            self._refresh_protocol_display()
            self._log(f"{result}\n")

    def _remove_protocol_step(self):
        index = self._selected_protocol_index()
        if index is None:
            return
        result = self.protocol_builder.remove_step(index)
        if result:
            self._refresh_protocol_display()
            self._log(f"{result}\n")

    def _clear_protocol(self):
        self.protocol_builder.clear_protocol()
        self._refresh_protocol_display()

    def _save_protocol(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".json", filetypes=(("JSON Files", "*.json"),))
        if path and self.protocol_builder.export_protocol(path):
            self._log(f"Protocol saved to {path}\n")

    def _load_protocol(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=(("JSON Files", "*.json"),))
        if path and self.protocol_builder.import_protocol(path):
            self._refresh_protocol_display()
            self._log(f"Protocol loaded from {path}\n")

    def queue_run_protocol(self):
        if not self._require_connection():
            return
        if not self.protocol_builder.protocol_list_configs:
            messagebox.showerror("Protocol Empty", "Add at least one protocol step.", parent=self.root)
            return
        self._run_in_background("Protocol Run", self._run_protocol)

    def _run_protocol(self):
        protocol = []
        for step in self.protocol_builder.protocol_list_configs:
            step_type = step.get("type")
            params = dict(step.get("params", {}))
            if step_type not in {"DCIV", "PULSE", "CV_CURRENT_PROBE", "DELAY", "LOG_MESSAGE"}:
                raise ValueError(f"Standalone GUI does not support protocol step '{step_type}'.")
            if step_type in {"DCIV", "PULSE", "CV_CURRENT_PROBE"}:
                params["align"] = False
                params["approach"] = False
            protocol.append({"type": step_type, "params": params})
        protocol_smus = self._ensure_protocol_smus(
            step["params"].get("smu_select", step["params"].get("smu"))
            for step in protocol
            if step["type"] in {"DCIV", "PULSE", "CV_CURRENT_PROBE"}
        )
        self.root.after(0, lambda: self._reset_live_plot("Protocol"))
        results = self.vipsa.run_protocol(
            protocol,
            sample_no=self._string_var("SAMPLE").get(),
            device_no=self._string_var("DEVICE").get(),
            save_directory=self._string_var("SAVE_DIR").get(),
            SMU=protocol_smus,
            stage=None,
            Zaber_x=None,
            Zaber_y=None,
            top_light=None,
            progress_callback=self._live_plot_callback,
        )
        self._log(f"Protocol results: {results}\n")

    def on_close(self):
        try:
            self.on_disconnect()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    StandaloneMeasurementTkApp().run()


if __name__ == "__main__":
    main()
