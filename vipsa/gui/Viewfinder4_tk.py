import csv
import io
import json
import os
import random
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2
import pandas as pd
from PIL import Image, ImageTk
import pyvisa
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from vipsa.analysis import Datahandling as dh
from vipsa.analysis.Vision import overlay
from vipsa.gui.ProtocolEditor_tk import ProtocolBuilderTk
from vipsa.gui.Testmaker import TestmakerApp
from vipsa.hardware.Movement import Light, Zaber, stage
from vipsa.hardware.Source_Measure_Unit import Keithley707B, KeithleySMU, KeysightSMU
from vipsa.workflows.Main4 import Vipsa_Methods


_BASE_DATA_DIR = os.path.join(
    os.path.expanduser("~"),
    "OneDrive - Nanyang Technological University",
    "ViPSA data folder",
)

if not os.path.isdir(_BASE_DATA_DIR):
    _BASE_DATA_DIR = os.path.expanduser("~")


def _default_csv_path(directory, preferred_names=()):
    if not os.path.isdir(directory):
        return ""
    for name in preferred_names:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    csv_files = sorted(name for name in os.listdir(directory) if name.lower().endswith(".csv"))
    if not csv_files:
        return ""
    return os.path.join(directory, csv_files[0])


DEFAULT_SAVE_DIRECTORY = _BASE_DATA_DIR
DEFAULT_SWEEP_DIRECTORY = os.path.join(_BASE_DATA_DIR, "sweep patterns")
DEFAULT_SWEEP_PATH = _default_csv_path(
    DEFAULT_SWEEP_DIRECTORY,
    preferred_names=("Sweep_2Dmems.csv", "tinyIV.csv", "Pulse20.csv"),
)
DEFAULT_PULSE_PATH = _default_csv_path(DEFAULT_SWEEP_DIRECTORY, preferred_names=("Pulse20.csv",))
SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".vipsa_control_center_settings.json")
CAMERA_SIZE = (640, 480)

BG_COLOR = "#1f2937"
PANEL_COLOR = "#111827"
SURFACE_COLOR = "#243447"
FIELD_COLOR = "#374151"
ACCENT_COLOR = "#f5c095"
ACCENT_HOVER = "#fe9400"
TEXT_COLOR = "#e5e7eb"
MUTED_TEXT = "#9ca3af"
ERROR_COLOR = "#ff0000"
SUCCESS_COLOR = "#28ec70"

DEFAULT_RUNTIME_SETTINGS = {
    "alignment": {
        "zaber_corr": True,
        "recheck": True,
    },
    "quick_approach": {
        "step_size": 1.0,
        "test_voltage": 0.1,
        "lower_threshold": 1e-9,
        "upper_threshold": 5e-8,
        "max_attempts": 50,
        "delay": 0.05,
    },
    "single_dciv": {
        "step_size": 0.5,
        "test_voltage": 0.05,
        "lower_threshold": 1e-11,
        "upper_threshold": 5e-11,
        "max_attempts": 50,
        "delay": 1.0,
        "current_autorange": False,
    },
    "single_pulse": {
        "step_size": 0.5,
        "test_voltage": 0.1,
        "lower_threshold": 1e-11,
        "upper_threshold": 5e-11,
        "max_attempts": 50,
        "delay": 1.0,
        "current_autorange": False,
    },
    "grid": {
        "step_size": 0.5,
        "test_voltage": 0.25,
        "lower_threshold": 1e-11,
        "upper_threshold": 5e-11,
        "max_attempts": 100,
        "delay": 1.0,
        "current_autorange": False,
    },
}


class RedirectText(io.TextIOBase):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def write(self, text):
        if text:
            self.callback(text)
        return len(text)

    def flush(self):
        return None


class ScrollableFrame(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=BG_COLOR)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.content.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self._window_id, width=event.width))


class VipsaGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("ViPSA Control Center (Tkinter)")
        self.master.geometry("1500x950")
        self.master.minsize(1280, 820)

        self.vipsa = Vipsa_Methods()
        self.protocol_builder = ProtocolBuilderTk(master, self.vipsa, logger=self._log)
        self.variables = {}
        self.widgets = {}
        self.camera_labels = []
        self.camera_thread = None
        self.camera_running = False
        self._cap = None
        self._camera_photo = None
        self.worker_thread = None
        self.is_equipment_connected = False
        self.switch = None
        self.smus = {}
        self.discovered_instruments = {}
        self.active_smu_label = None
        self.task_status = tk.StringVar(value="Idle")
        self.monitor_detected_text = tk.StringVar(value="Detected equipment: none")
        self.monitor_channels_text = tk.StringVar(value="707B closed channels: none")
        self.monitor_route_text = tk.StringVar(value="Active route: open")
        self.abort_event = threading.Event()
        self._monitor_after_id = None
        self.live_plot_title = "Live Sweep"
        self.live_plot_points = []
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._hardware_lock = threading.RLock()
        self._last_switch_channels = []
        self._last_raw_switch_channels = None
        self.testmaker_app = None

        self._configure_theme()
        self._build_layout()
        self._load_persisted_runtime_settings()

        sys.stdout = RedirectText(self._log)
        sys.stderr = RedirectText(self._log)
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_theme(self):
        self.master.configure(bg=BG_COLOR)
        style = ttk.Style(self.master)
        style.theme_use("clam")

        style.configure(".", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure("TFrame", background=BG_COLOR)
        style.configure("Card.TFrame", background=PANEL_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure("TLabelframe", background=BG_COLOR, foreground=TEXT_COLOR, borderwidth=1)
        style.configure("TLabelframe.Label", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure("TButton", background=SURFACE_COLOR, foreground=TEXT_COLOR, borderwidth=0, padding=(10, 6))
        style.map("TButton", background=[("active", ACCENT_HOVER), ("pressed", ACCENT_HOVER)])
        style.configure("Accent.TButton", background=ACCENT_COLOR, foreground=TEXT_COLOR)
        style.map("Accent.TButton", background=[("active", ACCENT_HOVER), ("pressed", ACCENT_HOVER)])
        style.configure(
            "TEntry",
            fieldbackground=FIELD_COLOR,
            foreground=TEXT_COLOR,
            insertcolor=TEXT_COLOR,
            bordercolor=SURFACE_COLOR,
        )
        style.configure(
            "TCombobox",
            fieldbackground=FIELD_COLOR,
            background=FIELD_COLOR,
            foreground=TEXT_COLOR,
            arrowsize=14,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", FIELD_COLOR)],
            foreground=[("readonly", TEXT_COLOR)],
            selectbackground=[("readonly", ACCENT_COLOR)],
            selectforeground=[("readonly", TEXT_COLOR)],
        )
        style.configure("TCheckbutton", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure("Vertical.TScrollbar", background=SURFACE_COLOR, troughcolor=PANEL_COLOR)
        style.configure("TPanedwindow", background=BG_COLOR)
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=SURFACE_COLOR, foreground=TEXT_COLOR, padding=(14, 8))
        style.map(
            "TNotebook.Tab",
            background=[("selected", ACCENT_COLOR), ("active", ACCENT_HOVER)],
            foreground=[("selected", TEXT_COLOR), ("active", TEXT_COLOR)],
        )
        style.configure("TSeparator", background=SURFACE_COLOR)

    def _register_string(self, key, default=""):
        var = tk.StringVar(value=str(default))
        self.variables[key] = var
        return var

    def _register_bool(self, key, default=False):
        existing = self.variables.get(key)
        if existing is not None:
            return existing
        var = tk.BooleanVar(value=bool(default))
        self.variables[key] = var
        return var

    def _get_value(self, key, default=""):
        var = self.variables.get(key)
        return var.get() if var is not None else default

    def _set_value(self, key, value):
        var = self.variables.get(key)
        if var is not None:
            var.set("" if value is None else value)

    def _collect_values(self):
        return {key: var.get() for key, var in self.variables.items()}

    def _append_log(self, text):
        if not hasattr(self, "log_text") or not self.log_text.winfo_exists():
            return
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _log(self, text):
        if not text:
            return
        if threading.current_thread() is threading.main_thread():
            self._append_log(text)
        else:
            try:
                self.master.after(0, self._append_log, text)
            except RuntimeError:
                pass

    def _set_status(self, text, color):
        label = self.widgets.get("connection_status_label")
        self._set_value("-CONNECTION_STATUS-", text)
        if label is not None:
            label.configure(fg=color)

    def _is_abort_requested(self):
        return self.abort_event.is_set()

    def _parse_closed_channels(self, raw_channels):
        text = str(raw_channels or "").strip().strip('"')
        if not text or text.lower() in {"none", "nil"}:
            return []

        cleaned = (
            text.replace("{", "")
            .replace("}", "")
            .replace("(", "")
            .replace(")", "")
            .replace('"', "")
            .replace(";", ",")
        )
        return [part.strip() for part in cleaned.split(",") if part.strip()]

    def _get_switch_channels(self):
        if self.switch is None:
            return [], None
        if not self._hardware_lock.acquire(blocking=False):
            return list(self._last_switch_channels), self._last_raw_switch_channels
        try:
            try:
                raw = self.switch.get_closed_channels()
            except Exception as exc:
                return list(self._last_switch_channels), f"error: {exc}"
            channels = self._parse_closed_channels(raw)
            self._last_switch_channels = list(channels)
            self._last_raw_switch_channels = raw
            return channels, raw
        finally:
            self._hardware_lock.release()

    def _identify_route(self, channels):
        closed = {channel.upper() for channel in channels}
        if {"1A01", "1B03"}.issubset(closed):
            return "Keithley"
        if {"1A02", "1B04"}.issubset(closed):
            return "Keysight"
        if closed:
            return "Custom"
        return "Open"

    def _safe_stop_hardware(self, log_message=True):
        if log_message:
            self._log("Issuing safe stop to stage, SMUs, and 707B...\n")

        stage = getattr(self.vipsa, "stage", None)
        if stage is not None and hasattr(stage, "abort_motion"):
            try:
                stage.abort_motion()
            except Exception as exc:
                self._log(f"Warning: stage abort raised an error: {exc}\n")

        smu_objects = []
        seen_ids = set()
        for smu in list(self.smus.values()) + [getattr(self.vipsa, "SMU", None)]:
            if smu is None or id(smu) in seen_ids:
                continue
            seen_ids.add(id(smu))
            smu_objects.append(smu)

        for smu in smu_objects:
            abort_measurement = getattr(smu, "abort_measurement", None)
            stop_output = getattr(smu, "stop_output", None)
            try:
                if callable(abort_measurement):
                    abort_measurement()
                elif callable(stop_output):
                    stop_output()
            except Exception as exc:
                self._log(f"Warning: could not safely stop SMU output: {exc}\n")

        if self.switch is not None:
            try:
                self.switch.open_all()
                self._last_switch_channels = []
                self._last_raw_switch_channels = "nil"
            except Exception as exc:
                self._log(f"Warning: could not open all 707B channels: {exc}\n")

        self._refresh_setup_monitor()

    def _restore_active_manual_route(self):
        if not self.is_equipment_connected or self.switch is None or self.active_smu_label is None:
            return
        smu = self.smus.get(self.active_smu_label)
        if smu is None:
            return
        connect_switch_path = getattr(smu, "connect_switch_path", None)
        if callable(connect_switch_path):
            connect_switch_path()
        channels, raw = self._get_switch_channels()
        self._last_switch_channels = list(channels)
        self._last_raw_switch_channels = raw
        self._refresh_setup_monitor()

    def _restore_arduino_stage_after_abort(self):
        stage_obj = getattr(self.vipsa, "stage", None)
        if stage_obj is None:
            return

        reconnect = getattr(stage_obj, "reconnect", None)
        if callable(reconnect):
            try:
                reconnect()
                self._log("Arduino stage reconnected after abort.\n")
            except Exception as exc:
                self._log(f"Warning: Arduino stage reconnect failed: {exc}\n")

    def _normalize_smu_label(self, label):
        text = str(label or "").strip().lower()
        if "keithley" in text or "2450" in text:
            return "keithley", "Keithley2450"
        if "keysight" in text or "b290" in text:
            return "keysight", "KeysightB2901BL"
        raise ValueError(f"Unknown SMU label: {label}")

    def _is_existing_file(self, path):
        return bool(path) and os.path.isfile(path)

    def _require_file(self, path, label):
        if self._is_existing_file(path):
            return True
        self._log(f"Error: {label} file not found: {path}\n")
        return False

    def _get_axis_position(self, axis):
        getter = getattr(axis, "get_position", None)
        if callable(getter):
            return getter()
        return float(getattr(axis, "position", 0.0))

    def _update_smu_stage_reference(self, smu):
        if smu is None:
            return
        set_stage_reference = getattr(smu, "set_stage_reference", None)
        if callable(set_stage_reference):
            try:
                set_stage_reference(getattr(self.vipsa, "stage", None))
            except Exception:
                pass

    def _reset_connection_state(self):
        self.is_equipment_connected = False
        self.switch = None
        self.smus = {}
        self.discovered_instruments = {}
        self.active_smu_label = None
        self._last_switch_channels = []
        self._last_raw_switch_channels = None
        self.vipsa.SMU = None
        self.vipsa.stage = None
        self.vipsa.Zaber = None
        self.vipsa.zaber_x = None
        self.vipsa.zaber_y = None
        self.vipsa.top_light = None
        self.vipsa.equipment = False

    def _cleanup_equipment_objects(self, smus=None, stage_obj=None, zaber_obj=None, top_light=None, switch_obj=None):
        if switch_obj is not None:
            try:
                switch_obj.open_all()
            except Exception:
                pass

        for smu in list(smus or []):
            try:
                close_session = getattr(smu, "close_session", None)
                if callable(close_session):
                    close_session()
            except Exception:
                pass

        if zaber_obj is not None:
            try:
                disconnect = getattr(zaber_obj, "disconnect", None)
                if callable(disconnect):
                    disconnect()
                else:
                    connection = getattr(zaber_obj, "connection", None)
                    if connection is not None:
                        connection.close()
            except Exception:
                pass

        if top_light is not None:
            try:
                control = getattr(top_light, "control_lights", None)
                if callable(control):
                    control("off")
            except Exception:
                pass
            try:
                disconnect = getattr(top_light, "disconnect", None)
                if callable(disconnect):
                    disconnect()
            except Exception:
                pass

        if stage_obj is not None:
            try:
                disconnect = getattr(stage_obj, "disconnect", None)
                if callable(disconnect):
                    disconnect()
            except Exception:
                pass

    def _schedule_saved_plot(self, plot_kind, saved_path, sample_id, device_id):
        if not saved_path:
            return

        def show():
            try:
                handler = dh.Data_Handler()
                if str(plot_kind).lower() == "pulse":
                    handler.show_pulse(saved_path, sample_id, device_id)
                else:
                    handler.show_plot(saved_path, sample_id, device_id)
            except Exception as exc:
                self._log(f"Warning: could not open {plot_kind} plot: {exc}\n")

        try:
            self.master.after(0, show)
        except RuntimeError:
            pass

    def _discover_visa_instruments(self):
        rm = pyvisa.ResourceManager()
        resources = list(rm.list_resources())
        discovered = {}

        for resource in resources:
            inst = None
            try:
                inst = rm.open_resource(resource)
                inst.timeout = 3000
                idn = inst.query("*IDN?").strip()
            except Exception:
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

        if "switch" not in discovered:
            known = {value for key, value in discovered.items() if key in {"keithley", "keysight"}}
            remaining = [resource for resource in resources if resource not in known]
            if len(remaining) == 1:
                discovered["switch"] = remaining[0]

        return discovered

    def _build_smu(self, label):
        key, _ = self._normalize_smu_label(label)
        if key not in self.discovered_instruments:
            raise RuntimeError(f"No detected VISA address for '{label}'.")
        if self.switch is None:
            raise RuntimeError("707B switch is not connected.")

        address = self.discovered_instruments[key]
        if key == "keithley":
            smu = KeithleySMU(device_no=0, address=address, switch=self.switch, switch_channel="keithley")
        else:
            smu = KeysightSMU(device_no=0, address=address, switch=self.switch, switch_channel="keysight")
        self._update_smu_stage_reference(smu)
        return smu

    def _connect_real_equipment(self, selected_smu):
        if self.is_equipment_connected:
            self._disconnect_all(log_message=False)

        discovered = self._discover_visa_instruments()
        missing = [name for name in ("switch", "keithley", "keysight") if name not in discovered]
        if missing:
            raise RuntimeError(f"Could not auto-detect: {', '.join(missing)}")

        self.discovered_instruments = discovered
        self._log(f"Detected 707B switch at {discovered['switch']}\n")
        self._log(f"Detected Keithley SMU at {discovered['keithley']}\n")
        self._log(f"Detected Keysight SMU at {discovered['keysight']}\n")

        top_light = None
        stage_obj = None
        zaber_obj = None
        switch_obj = None
        try:
            arduino_port = self._get_value("-ARDUINO_PORT-", "COM5")
            zaber_port = self._get_value("-ZABER_PORT-", "COM7")
            arduino_baud = int(float(self._get_value("-ARDUINO_BAUD-", "115200")))
            arduino_scale = float(self._get_value("-ARDUINO_SCALE-", "1"))

            top_light = Light()
            if getattr(top_light, "ser_lights", None) is None:
                raise RuntimeError("Could not connect lights.")

            stage_obj = stage(arduino_port, arduino_baud, arduino_scale)
            if getattr(stage_obj, "ser", None) is None:
                raise RuntimeError(f"Could not connect Arduino stage on {arduino_port}.")

            zaber_obj = Zaber(zaber_port)
            zaber_x, zaber_y = zaber_obj.get_devices()

            switch_obj = Keithley707B(address=discovered["switch"], slot=1)
            switch_obj.open_all()

            self.vipsa.top_light = top_light
            self.vipsa.stage = stage_obj
            self.vipsa.Zaber = zaber_obj
            self.vipsa.zaber_x = zaber_x
            self.vipsa.zaber_y = zaber_y
            self.vipsa.SMU_name = selected_smu
            self.vipsa.equipment = True

            self.switch = switch_obj
            self.smus = {}
            self.active_smu_label = None
            self.is_equipment_connected = True

            self._set_manual_active_smu(selected_smu, force_reconnect=True)
            self._set_status(f"Connected ({selected_smu})", SUCCESS_COLOR)
            self._log(
                f"Connected real hardware using Arduino={arduino_port}, Zaber={zaber_port}, startup SMU={selected_smu}\n"
            )
            self._refresh_setup_monitor()
        except Exception:
            self._cleanup_equipment_objects(stage_obj=stage_obj, zaber_obj=zaber_obj, top_light=top_light, switch_obj=switch_obj)
            self._reset_connection_state()
            raise

    def _disconnect_smu_object(self, label):
        key, _ = self._normalize_smu_label(label)
        smu = self.smus.pop(key, None)
        if smu is None:
            return
        try:
            close_session = getattr(smu, "close_session", None)
            if callable(close_session):
                close_session()
        except Exception as exc:
            self._log(f"Warning: could not fully close {label}: {exc}\n")

    def _set_manual_active_smu(self, label, force_reconnect=False):
        with self._hardware_lock:
            key, display = self._normalize_smu_label(label)
            if not self.discovered_instruments or self.switch is None:
                raise RuntimeError("Equipment must be connected first.")

            if self.active_smu_label == key and key in self.smus and not force_reconnect:
                self.vipsa.SMU = self.smus[key]
                self._update_smu_stage_reference(self.smus[key])
                return self.smus[key]

            self.switch.open_all()
            self._last_switch_channels = []
            self._last_raw_switch_channels = "nil"

            if self.active_smu_label is not None and self.active_smu_label != key:
                self._disconnect_smu_object(self.active_smu_label)

            if key not in self.smus or force_reconnect:
                self._disconnect_smu_object(key)
                self.smus[key] = self._build_smu(display)

            self.active_smu_label = key
            self.vipsa.SMU = self.smus[key]
            self._update_smu_stage_reference(self.smus[key])
            connect_switch_path = getattr(self.smus[key], "connect_switch_path", None)
            if callable(connect_switch_path):
                connect_switch_path()
            self._set_value("-ACTIVE_SMU-", display)
            self._set_value("-SMU_SELECT-", display)
            self._set_status(f"Connected ({display})", SUCCESS_COLOR)
            self._log(f"Active manual SMU switched to {display} at {self.discovered_instruments[key]}\n")
            self._refresh_setup_monitor()
            return self.smus[key]

    def _ensure_protocol_smus(self, protocol_steps):
        protocol_smus = {}
        for step in protocol_steps:
            params = step.get("params", {})
            label = params.get("smu", params.get("smu_type", params.get("smu_select", None)))
            if label is None:
                continue
            key, display = self._normalize_smu_label(label)
            if key not in self.smus:
                self.smus[key] = self._build_smu(display)
            protocol_smus[key] = self.smus[key]
        return protocol_smus

    def _set_protocol_steps(self, protocol_steps, source_label="manual"):
        ok, message = self.vipsa.validate_protocol(protocol_steps)
        if not ok:
            raise ValueError(message)
        self.protocol_builder.protocol_list_configs = list(protocol_steps)
        self._refresh_protocol_display()
        self._log(f"Protocol synced from {source_label}. Steps: {len(protocol_steps)}\n")

    def _on_testmaker_protocol(self, protocol_steps, csv_path, generated):
        del csv_path
        source_name = "Testmaker"
        try:
            test_name = generated.get("test", {}).get("name")
            if test_name:
                source_name = f"Testmaker ({test_name})"
        except Exception:
            pass
        self._set_protocol_steps(protocol_steps, source_label=source_name)
        try:
            self.notebook.select(self.notebook.tabs()[-1])
        except Exception:
            pass

    def _on_testmaker_closed(self):
        self.testmaker_app = None

    def _open_testmaker(self):
        if self.testmaker_app is not None:
            try:
                self.testmaker_app.root.lift()
                self.testmaker_app.root.focus_force()
                return
            except Exception:
                self.testmaker_app = None
        self.testmaker_app = TestmakerApp(
            parent=self.master,
            protocol_callback=self._on_testmaker_protocol,
            close_callback=self._on_testmaker_closed,
        )

    def _set_task_status(self, text):
        self.task_status.set(text)

    def _queue_connect_all(self):
        self._run_in_background("Connect equipment", self._connect_all)

    def _queue_disconnect_all(self):
        self._run_in_background("Disconnect equipment", self._disconnect_all)

    def _queue_switch_instrument(self):
        self._run_in_background("Switch instrument", self._switch_instrument)

    def _run_in_background(self, label, func, *args):
        if self.worker_thread and self.worker_thread.is_alive():
            self._log(f"A background task is already running: {self.task_status.get()}\n")
            return False

        self.abort_event.clear()

        def runner():
            try:
                self.master.after(0, self._set_task_status, f"Running: {label}")
                self.master.after(0, self._reset_live_plot, label)
                self._log(f"{label} started.\n")
                with self._hardware_lock:
                    func(*args)
                if self._is_abort_requested():
                    self._log(f"{label} aborted.\n")
                else:
                    self._log(f"{label} finished.\n")
            except Exception as exc:
                self._log(f"{label} failed: {exc}\n")
            finally:
                try:
                    with self._hardware_lock:
                        if self.is_equipment_connected:
                            if self._is_abort_requested():
                                self._safe_stop_hardware(log_message=False)
                                self._restore_arduino_stage_after_abort()
                            self._restore_active_manual_route()
                    self.master.after(0, self._refresh_setup_monitor)
                    self.abort_event.clear()
                    self.master.after(0, self._set_task_status, "Idle")
                except RuntimeError:
                    pass

        self.worker_thread = threading.Thread(target=runner, daemon=True)
        self.worker_thread.start()
        return True

    def _choose_file(self, key, filetypes=None):
        path = filedialog.askopenfilename(parent=self.master, filetypes=filetypes or [("All Files", "*.*")])
        if path:
            self._set_value(key, path)

    def _choose_directory(self, key):
        path = filedialog.askdirectory(parent=self.master)
        if path:
            self._set_value(key, path)

    def _add_entry_row(self, parent, label, key, default="", width=14, browse=None, filetypes=None, persist_on_focus_out=False):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=18).pack(side="left")
        entry = ttk.Entry(row, textvariable=self._register_string(key, default), width=width)
        entry.pack(side="left", fill="x", expand=True)
        self.widgets[key] = entry
        if persist_on_focus_out:
            entry.bind("<FocusOut>", lambda _event: self._persist_runtime_settings(notify=False))
            entry.bind("<Return>", lambda _event: self._persist_runtime_settings(notify=False))
        if browse == "file":
            ttk.Button(row, text="Browse", command=lambda: self._choose_file(key, filetypes=filetypes)).pack(side="left", padx=(6, 0))
        elif browse == "dir":
            ttk.Button(row, text="Browse", command=lambda: self._choose_directory(key)).pack(side="left", padx=(6, 0))
        return entry

    def _add_checkbox(self, parent, text, key, default=False, command=None):
        check = ttk.Checkbutton(parent, text=text, variable=self._register_bool(key, default), command=command)
        check.pack(anchor="w", pady=2)
        self.widgets[key] = check
        return check

    def _add_section(self, parent, title):
        section = ttk.LabelFrame(parent, text=title, padding=8)
        section.pack(fill="x", pady=6)
        return section

    def _default_runtime_settings(self):
        return json.loads(json.dumps(DEFAULT_RUNTIME_SETTINGS))

    def _merge_runtime_settings(self, defaults, loaded):
        if not isinstance(loaded, dict):
            return defaults
        merged = json.loads(json.dumps(defaults))
        for group, values in loaded.items():
            if group not in merged or not isinstance(values, dict):
                continue
            for key, value in values.items():
                if key in merged[group]:
                    merged[group][key] = value
        return merged

    def _apply_runtime_settings_to_ui(self, settings):
        self._set_value("-SET_ALIGN_ZABER_CORR-", bool(settings["alignment"]["zaber_corr"]))
        self._set_value("-SET_ALIGN_RECHECK-", bool(settings["alignment"]["recheck"]))

        mapping = {
            "-SET_QUICK": settings["quick_approach"],
            "-SET_SINGLE_DCIV": settings["single_dciv"],
            "-SET_SINGLE_PULSE": settings["single_pulse"],
            "-SET_GRID": settings["grid"],
        }
        for prefix, values in mapping.items():
            self._set_value(f"{prefix}_STEP_SIZE", values["step_size"])
            self._set_value(f"{prefix}_TEST_VOLTAGE", values["test_voltage"])
            self._set_value(f"{prefix}_LOWER_THRESHOLD", values["lower_threshold"])
            self._set_value(f"{prefix}_UPPER_THRESHOLD", values["upper_threshold"])
            self._set_value(f"{prefix}_MAX_ATTEMPTS", values["max_attempts"])
            self._set_value(f"{prefix}_DELAY", values["delay"])
            if "current_autorange" in values:
                self._set_value(f"{prefix}_AUTORANGE-", bool(values["current_autorange"]))

    def _load_persisted_runtime_settings(self):
        settings = self._default_runtime_settings()
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                settings = self._merge_runtime_settings(settings, loaded)
            except Exception as exc:
                self._log(f"Warning: could not load settings file: {exc}\n")
        self._apply_runtime_settings_to_ui(settings)
        if not os.path.exists(SETTINGS_FILE):
            self._persist_runtime_settings(notify=False)

    def _persist_runtime_settings(self, notify=True):
        try:
            settings = self._build_runtime_settings()
            os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
            with open(SETTINGS_FILE, "w", encoding="utf-8") as handle:
                json.dump(settings, handle, indent=2)
            if notify:
                self._log(f"Settings saved to {SETTINGS_FILE}\n")
            return True
        except Exception as exc:
            if notify:
                self._log(f"Could not save settings: {exc}\n")
            return False

    def _restore_default_runtime_settings(self):
        self._apply_runtime_settings_to_ui(self._default_runtime_settings())
        self._persist_runtime_settings(notify=True)
        self._log("Runtime settings restored to defaults.\n")

    def _get_runtime_bool_setting(self, key, fallback):
        value = self._get_value(key, fallback)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(fallback)

    def _get_runtime_float_setting(self, key, fallback):
        try:
            return float(self._get_value(key, fallback))
        except (TypeError, ValueError):
            return float(fallback)

    def _get_runtime_int_setting(self, key, fallback):
        try:
            return int(float(self._get_value(key, fallback)))
        except (TypeError, ValueError):
            return int(fallback)

    def _get_runtime_contact_settings(self, prefix, defaults):
        return {
            "step_size": self._get_runtime_float_setting(f"{prefix}_STEP_SIZE", defaults["step_size"]),
            "test_voltage": self._get_runtime_float_setting(f"{prefix}_TEST_VOLTAGE", defaults["test_voltage"]),
            "lower_threshold": self._get_runtime_float_setting(f"{prefix}_LOWER_THRESHOLD", defaults["lower_threshold"]),
            "upper_threshold": self._get_runtime_float_setting(f"{prefix}_UPPER_THRESHOLD", defaults["upper_threshold"]),
            "max_attempts": self._get_runtime_int_setting(f"{prefix}_MAX_ATTEMPTS", defaults["max_attempts"]),
            "delay": self._get_runtime_float_setting(f"{prefix}_DELAY", defaults["delay"]),
        }

    def _get_runtime_measurement_settings(self, prefix, defaults):
        settings = self._get_runtime_contact_settings(prefix, defaults)
        if "current_autorange" in defaults:
            settings["current_autorange"] = self._get_runtime_bool_setting(
                f"{prefix}_AUTORANGE-",
                defaults["current_autorange"],
            )
        return settings

    def _build_runtime_settings(self):
        defaults = self._default_runtime_settings()
        return {
            "alignment": {
                "zaber_corr": self._get_runtime_bool_setting("-SET_ALIGN_ZABER_CORR-", defaults["alignment"]["zaber_corr"]),
                "recheck": self._get_runtime_bool_setting("-SET_ALIGN_RECHECK-", defaults["alignment"]["recheck"]),
            },
            "quick_approach": self._get_runtime_contact_settings("-SET_QUICK", defaults["quick_approach"]),
            "single_dciv": self._get_runtime_measurement_settings("-SET_SINGLE_DCIV", defaults["single_dciv"]),
            "single_pulse": self._get_runtime_measurement_settings("-SET_SINGLE_PULSE", defaults["single_pulse"]),
            "grid": self._get_runtime_measurement_settings("-SET_GRID", defaults["grid"]),
        }

    def _create_camera_block(self, parent, key_prefix):
        block = ttk.LabelFrame(parent, text="Camera Feed", padding=8)
        block.pack(fill="x", anchor="n")
        preview_frame = ttk.Frame(block, width=CAMERA_SIZE[0], height=CAMERA_SIZE[1])
        preview_frame.pack(anchor="n")
        preview_frame.pack_propagate(False)
        image_label = tk.Label(preview_frame, bg="black", fg=TEXT_COLOR, text="Camera stopped", relief="flat")
        image_label.pack(fill="both", expand=True)
        self.widgets[f"{key_prefix}_camera_label"] = image_label
        self.camera_labels.append(image_label)
        controls = ttk.Frame(block)
        controls.pack(fill="x", pady=(8, 0))
        ttk.Button(controls, text="Start Camera", command=self._start_camera_thread, style="Accent.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Stop Camera", command=self._stop_camera_thread).pack(side="left")

    def _build_layout(self):
        main_pane = ttk.Panedwindow(self.master, orient=tk.HORIZONTAL)
        main_pane.pack(fill="both", expand=True)
        left_panel = ttk.Frame(main_pane, padding=8)
        right_panel = ttk.Frame(main_pane, padding=8)
        main_pane.add(left_panel, weight=5)
        main_pane.add(right_panel, weight=2)

        self.notebook = ttk.Notebook(left_panel)
        self.notebook.pack(fill="both", expand=True)
        self._create_movement_tab()
        self._create_settings_tab()
        self._create_grid_tab()
        self._create_measure_tab()
        self._create_protocol_tab()
        self._create_monitor_panel(right_panel)
        self._create_live_plot_panel(right_panel)
        self._create_log_panel(right_panel)
        self._schedule_monitor_refresh()

    def _create_movement_tab(self):
        frame = ttk.Frame(self.notebook, padding=8)
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(0, weight=1)
        self.notebook.add(frame, text="Movement & Connections")

        left = ttk.Frame(frame)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._create_camera_block(left, "move")

        right = ScrollableFrame(frame)
        right.grid(row=0, column=1, sticky="nsew")
        controls = right.content

        connections = self._add_section(controls, "Connections")
        ttk.Button(connections, text="Connect All Equipments", command=self._queue_connect_all).pack(fill="x", pady=2)
        ttk.Button(connections, text="Disconnect All", command=self._queue_disconnect_all).pack(fill="x", pady=2)
        smu_switch_row = ttk.Frame(connections)
        smu_switch_row.pack(fill="x", pady=(4, 0))
        ttk.Label(smu_switch_row, text="Active SMU", width=18).pack(side="left")
        ttk.Combobox(
            smu_switch_row,
            textvariable=self._register_string("-ACTIVE_SMU-", "Keithley2450"),
            values=("Keithley2450", "KeysightB2901BL"),
            state="readonly",
            width=18,
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(smu_switch_row, text="Switch Instrument", command=self._queue_switch_instrument).pack(side="left", padx=(6, 0))
        status_row = ttk.Frame(connections)
        status_row.pack(fill="x", pady=(4, 0))
        ttk.Label(status_row, text="Status:", width=18).pack(side="left")
        self._register_string("-CONNECTION_STATUS-", "Disconnected")
        status_label = tk.Label(status_row, textvariable=self.variables["-CONNECTION_STATUS-"], fg=ERROR_COLOR, bg=BG_COLOR)
        status_label.pack(side="left", anchor="w")
        self.widgets["connection_status_label"] = status_label

        arduino = self._add_section(controls, "Arduino Stage (OpenFlexure)")
        ports = ttk.Frame(arduino)
        ports.pack(fill="x", pady=2)
        ttk.Label(ports, text="Port").pack(side="left")
        ttk.Entry(ports, textvariable=self._register_string("-ARDUINO_PORT-", "COM5"), width=10).pack(side="left", padx=(4, 12))
        ttk.Label(ports, text="Baud").pack(side="left")
        ttk.Entry(ports, textvariable=self._register_string("-ARDUINO_BAUD-", "115200"), width=10).pack(side="left", padx=(4, 12))
        ttk.Label(ports, text="Scale").pack(side="left")
        ttk.Entry(ports, textvariable=self._register_string("-ARDUINO_SCALE-", "1"), width=10).pack(side="left", padx=(4, 0))
        for axis in ("X", "Y", "Z"):
            row = ttk.Frame(arduino)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"{axis} Steps", width=18).pack(side="left")
            ttk.Entry(row, textvariable=self._register_string(f"-{axis}_STEPS-", "10"), width=10).pack(side="left", padx=(0, 8))
            ttk.Button(row, text=f"{axis}+", width=4, command=lambda a=axis: self._handle_arduino_move(a, 1)).pack(side="left")
            ttk.Button(row, text=f"{axis}-", width=4, command=lambda a=axis: self._handle_arduino_move(a, -1)).pack(side="left", padx=(6, 0))
        zero_row = ttk.Frame(arduino)
        zero_row.pack(fill="x", pady=(6, 0))
        ttk.Button(zero_row, text="Set Zero", command=self._set_zero).pack(side="left")
        ttk.Button(zero_row, text="Go to Zero", command=self._go_zero).pack(side="left", padx=(6, 0))

        zaber = self._add_section(controls, "Zaber Stage")
        self._add_entry_row(zaber, "Port", "-ZABER_PORT-", "COM4", width=16)
        for axis in ("X", "Y"):
            row = ttk.Frame(zaber)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"{axis} Distance", width=18).pack(side="left")
            ttk.Entry(row, textvariable=self._register_string(f"-ZABER_{axis}_DIST-", "1000"), width=10).pack(side="left", padx=(0, 8))
            ttk.Button(row, text=f"{axis}+", width=4, command=lambda a=axis: self._handle_zaber_move(a, 1)).pack(side="left")
            ttk.Button(row, text=f"{axis}-", width=4, command=lambda a=axis: self._handle_zaber_move(a, -1)).pack(side="left", padx=(6, 0))

        lights = self._add_section(controls, "Lights")
        ttk.Button(lights, text="Turn On Lights", command=lambda: self._control_lights("on")).pack(fill="x", pady=2)
        ttk.Button(lights, text="Turn Off Lights", command=lambda: self._control_lights("off")).pack(fill="x", pady=2)

        smu = self._add_section(controls, "SMU")
        row = ttk.Frame(smu)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Manual SMU", width=18).pack(side="left")
        ttk.Combobox(
            row,
            textvariable=self._register_string("-SMU_SELECT-", "Keithley2450"),
            values=("Keithley2450", "KeysightB2901BL"),
            state="disabled",
            width=18,
        ).pack(side="left", fill="x", expand=True)
        actions = ttk.Frame(smu)
        actions.pack(fill="x", pady=(6, 0))
        ttk.Button(actions, text="Quick Align", command=self._queue_quick_align).pack(side="left")
        ttk.Button(actions, text="Quick Approach", command=self._queue_quick_approach).pack(side="left", padx=(6, 0))

    def _create_settings_tab(self):
        frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(frame, text="Settings")
        scroll = ScrollableFrame(frame)
        scroll.pack(fill="both", expand=True)
        content = scroll.content

        intro = self._add_section(content, "Runtime Defaults")
        ttk.Label(
            intro,
            text=f"These values replace the hidden baseline numbers used for quick actions, single measurements, grid runs, and protocol steps. They are saved to {SETTINGS_FILE} and reloaded on restart.",
            wraplength=760,
            justify="left",
        ).pack(anchor="w")
        actions = ttk.Frame(intro)
        actions.pack(anchor="w", pady=(8, 0))
        ttk.Button(actions, text="Save Settings", command=lambda: self._persist_runtime_settings(notify=True)).pack(side="left")
        ttk.Button(actions, text="Restore Defaults", command=self._restore_default_runtime_settings).pack(side="left", padx=(6, 0))

        alignment = self._add_section(content, "Alignment Defaults")
        self._add_checkbox(
            alignment,
            "Use Zaber correction during alignment",
            "-SET_ALIGN_ZABER_CORR-",
            default=DEFAULT_RUNTIME_SETTINGS["alignment"]["zaber_corr"],
            command=lambda: self._persist_runtime_settings(notify=False),
        )
        self._add_checkbox(
            alignment,
            "Recheck after alignment correction",
            "-SET_ALIGN_RECHECK-",
            default=DEFAULT_RUNTIME_SETTINGS["alignment"]["recheck"],
            command=lambda: self._persist_runtime_settings(notify=False),
        )

        quick = self._add_section(content, "Quick Approach Defaults")
        self._add_entry_row(quick, "Step Size", "-SET_QUICK_STEP_SIZE", str(DEFAULT_RUNTIME_SETTINGS["quick_approach"]["step_size"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(quick, "Probe Voltage (V)", "-SET_QUICK_TEST_VOLTAGE", str(DEFAULT_RUNTIME_SETTINGS["quick_approach"]["test_voltage"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(quick, "Lower Threshold (A)", "-SET_QUICK_LOWER_THRESHOLD", str(DEFAULT_RUNTIME_SETTINGS["quick_approach"]["lower_threshold"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(quick, "Upper Threshold (A)", "-SET_QUICK_UPPER_THRESHOLD", str(DEFAULT_RUNTIME_SETTINGS["quick_approach"]["upper_threshold"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(quick, "Max Attempts", "-SET_QUICK_MAX_ATTEMPTS", str(DEFAULT_RUNTIME_SETTINGS["quick_approach"]["max_attempts"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(quick, "Settle Delay (s)", "-SET_QUICK_DELAY", str(DEFAULT_RUNTIME_SETTINGS["quick_approach"]["delay"]), width=18, persist_on_focus_out=True)

        single_dciv = self._add_section(content, "Single DCIV Auto-Contact Defaults")
        self._add_entry_row(single_dciv, "Step Size", "-SET_SINGLE_DCIV_STEP_SIZE", str(DEFAULT_RUNTIME_SETTINGS["single_dciv"]["step_size"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(single_dciv, "Probe Voltage (V)", "-SET_SINGLE_DCIV_TEST_VOLTAGE", str(DEFAULT_RUNTIME_SETTINGS["single_dciv"]["test_voltage"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(single_dciv, "Lower Threshold (A)", "-SET_SINGLE_DCIV_LOWER_THRESHOLD", str(DEFAULT_RUNTIME_SETTINGS["single_dciv"]["lower_threshold"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(single_dciv, "Upper Threshold (A)", "-SET_SINGLE_DCIV_UPPER_THRESHOLD", str(DEFAULT_RUNTIME_SETTINGS["single_dciv"]["upper_threshold"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(single_dciv, "Max Attempts", "-SET_SINGLE_DCIV_MAX_ATTEMPTS", str(DEFAULT_RUNTIME_SETTINGS["single_dciv"]["max_attempts"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(single_dciv, "Settle Delay (s)", "-SET_SINGLE_DCIV_DELAY", str(DEFAULT_RUNTIME_SETTINGS["single_dciv"]["delay"]), width=18, persist_on_focus_out=True)
        self._add_checkbox(single_dciv, "Use Current Autorange", "-SET_SINGLE_DCIV_AUTORANGE-", default=DEFAULT_RUNTIME_SETTINGS["single_dciv"]["current_autorange"], command=lambda: self._persist_runtime_settings(notify=False))

        single_pulse = self._add_section(content, "Single Pulse Auto-Contact Defaults")
        self._add_entry_row(single_pulse, "Step Size", "-SET_SINGLE_PULSE_STEP_SIZE", str(DEFAULT_RUNTIME_SETTINGS["single_pulse"]["step_size"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(single_pulse, "Probe Voltage (V)", "-SET_SINGLE_PULSE_TEST_VOLTAGE", str(DEFAULT_RUNTIME_SETTINGS["single_pulse"]["test_voltage"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(single_pulse, "Lower Threshold (A)", "-SET_SINGLE_PULSE_LOWER_THRESHOLD", str(DEFAULT_RUNTIME_SETTINGS["single_pulse"]["lower_threshold"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(single_pulse, "Upper Threshold (A)", "-SET_SINGLE_PULSE_UPPER_THRESHOLD", str(DEFAULT_RUNTIME_SETTINGS["single_pulse"]["upper_threshold"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(single_pulse, "Max Attempts", "-SET_SINGLE_PULSE_MAX_ATTEMPTS", str(DEFAULT_RUNTIME_SETTINGS["single_pulse"]["max_attempts"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(single_pulse, "Settle Delay (s)", "-SET_SINGLE_PULSE_DELAY", str(DEFAULT_RUNTIME_SETTINGS["single_pulse"]["delay"]), width=18, persist_on_focus_out=True)
        self._add_checkbox(single_pulse, "Use Current Autorange", "-SET_SINGLE_PULSE_AUTORANGE-", default=DEFAULT_RUNTIME_SETTINGS["single_pulse"]["current_autorange"], command=lambda: self._persist_runtime_settings(notify=False))

        grid = self._add_section(content, "Grid Auto-Contact Defaults")
        self._add_entry_row(grid, "Step Size", "-SET_GRID_STEP_SIZE", str(DEFAULT_RUNTIME_SETTINGS["grid"]["step_size"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(grid, "Probe Voltage (V)", "-SET_GRID_TEST_VOLTAGE", str(DEFAULT_RUNTIME_SETTINGS["grid"]["test_voltage"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(grid, "Lower Threshold (A)", "-SET_GRID_LOWER_THRESHOLD", str(DEFAULT_RUNTIME_SETTINGS["grid"]["lower_threshold"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(grid, "Upper Threshold (A)", "-SET_GRID_UPPER_THRESHOLD", str(DEFAULT_RUNTIME_SETTINGS["grid"]["upper_threshold"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(grid, "Max Attempts", "-SET_GRID_MAX_ATTEMPTS", str(DEFAULT_RUNTIME_SETTINGS["grid"]["max_attempts"]), width=18, persist_on_focus_out=True)
        self._add_entry_row(grid, "Settle Delay (s)", "-SET_GRID_DELAY", str(DEFAULT_RUNTIME_SETTINGS["grid"]["delay"]), width=18, persist_on_focus_out=True)
        self._add_checkbox(grid, "Use Current Autorange", "-SET_GRID_AUTORANGE-", default=DEFAULT_RUNTIME_SETTINGS["grid"]["current_autorange"], command=lambda: self._persist_runtime_settings(notify=False))

    def _create_grid_tab(self):
        frame = ttk.Frame(self.notebook, padding=8)
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(0, weight=1)
        self.notebook.add(frame, text="Grid Creation & Measurement")

        left = ttk.Frame(frame)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._create_camera_block(left, "grid")

        right = ScrollableFrame(frame)
        right.grid(row=0, column=1, sticky="nsew")
        controls = right.content

        create_section = self._add_section(controls, "Grid Creation")
        self._add_entry_row(create_section, "X Distance", "-X_DIST-", "", width=14)
        self._add_entry_row(create_section, "Y Distance", "-Y_DIST-", "", width=14)
        self._add_entry_row(create_section, "Rows", "-ROWS-", "", width=14)
        self._add_entry_row(create_section, "Columns", "-COLS-", "", width=14)
        self._add_entry_row(create_section, "Grid Save Path", "-GRID_PATH-", DEFAULT_SAVE_DIRECTORY, width=36, browse="dir")
        ttk.Button(create_section, text="Create Grid CSV", command=lambda: self._create_grid_csv(self._collect_values())).pack(fill="x", pady=(6, 0))

        run_section = self._add_section(controls, "Grid Measurement")
        self._add_entry_row(run_section, "Grid File", "-GRID_MEAS_PATH-", "", width=36, browse="file", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        self._add_entry_row(run_section, "Sample ID", "-GRID_SAMPLE_ID-", "sample", width=18)
        self._add_entry_row(run_section, "Start Device #", "-GRID_STARTPOINT-", "1", width=10)
        self._add_entry_row(run_section, "Skip Devices", "-GRID_SKIP-", "1", width=10)
        self._add_checkbox(run_section, "Randomize Order", "-GRID_RANDOMIZE-", default=False)
        ttk.Label(run_section, text="Use the protocol defined in the Protocol Builder tab for protocol runs.").pack(anchor="w", pady=(2, 4))
        ttk.Button(run_section, text="Run Grid Measurement (DCIV)", command=lambda: self._queue_grid_measurement("DCIV")).pack(fill="x", pady=2)
        ttk.Button(run_section, text="Run Grid Measurement (Pulse)", command=lambda: self._queue_grid_measurement("PULSE")).pack(fill="x", pady=2)
        ttk.Button(run_section, text="Run Grid Measurement (Protocol)", command=lambda: self._queue_grid_measurement("PROTOCOL")).pack(fill="x", pady=2)

        params = self._add_section(controls, "Grid Parameters")
        self._add_entry_row(params, "Pos Comp", "-GRID_POS_COMP-", "0.001", width=14)
        self._add_entry_row(params, "Neg Comp", "-GRID_NEG_COMP-", "0.01", width=14)
        self._add_entry_row(params, "Pulse Comp", "-GRID_PULSE_COMP-", "0.01", width=14)
        self._add_entry_row(params, "Sweep Path", "-GRID_SWEEP_PATH-", DEFAULT_SWEEP_PATH, width=36, browse="file", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        self._add_entry_row(params, "Pulse Path", "-GRID_PULSE_PATH-", DEFAULT_PULSE_PATH, width=36, browse="file", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        self._add_entry_row(params, "Sweep Delay", "-GRID_SWEEP_DELAY-", "0.0001", width=14)
        self._add_entry_row(params, "Pulse Width", "-GRID_PULSE_WIDTH-", "0.001", width=14)
        self._add_checkbox(params, "Use 4-way split for Grid DCIV", "-GRID_USE_4WAY_SPLIT-", default=True)
        self._add_checkbox(params, "Include HRS/LRS Probe During Grid DCIV", "-GRID_INCLUDE_READ_PROBE-", default=True)
        self._add_checkbox(params, "Use Current Autorange", "-SET_GRID_AUTORANGE-", default=DEFAULT_RUNTIME_SETTINGS["grid"]["current_autorange"], command=lambda: self._persist_runtime_settings(notify=False))

    def _create_measure_tab(self):
        frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(frame, text="Single Measurement")
        scroll = ScrollableFrame(frame)
        scroll.pack(fill="both", expand=True)
        content = scroll.content

        target = self._add_section(content, "Single Measurement Target")
        self._add_entry_row(target, "Sample ID", "-MEAS_SAMPLE_ID-", "sample", width=18)
        self._add_entry_row(target, "Device #", "-MEAS_DEVICE_ID-", "0", width=18)
        self._add_entry_row(target, "Save Folder", "-MEAS_SAVE_FOLDER-", DEFAULT_SAVE_DIRECTORY, width=40, browse="dir")
        self._add_checkbox(target, "Align First", "-MEAS_ALIGN-", default=False)
        self._add_checkbox(target, "Approach First", "-MEAS_APPROACH-", default=False)
        self._add_entry_row(target, "Wait Between Runs (s)", "-WAIT_TIME-", "0", width=18)

        resistance = self._add_section(content, "Resistance Check")
        self._add_entry_row(resistance, "Checking Voltage (V)", "-MEAS_RES_VOLTAGE-", "0.01", width=18)
        ttk.Button(resistance, text="Measure Resistance", command=self._queue_measure_resistance).pack(fill="x", pady=(6, 0))

        dciv = self._add_section(content, "DCIV Sweep Parameters")
        self._add_entry_row(dciv, "Sweep Path", "-MEAS_SWEEP_PATH-", DEFAULT_SWEEP_PATH, width=40, browse="file", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        self._add_entry_row(dciv, "Pos Comp (A)", "-MEAS_POS_COMP-", "0.001", width=18)
        self._add_entry_row(dciv, "Neg Comp (A)", "-MEAS_NEG_COMP-", "0.01", width=18)
        self._add_entry_row(dciv, "Sweep Delay (s)", "-MEAS_SWEEP_DELAY-", "0.0001", width=18)
        self._add_checkbox(dciv, "Use 4-way split", "-MEAS_USE_4WAY_SPLIT-", default=True)
        self._add_checkbox(dciv, "Use Current Autorange", "-SET_SINGLE_DCIV_AUTORANGE-", default=DEFAULT_RUNTIME_SETTINGS["single_dciv"]["current_autorange"], command=lambda: self._persist_runtime_settings(notify=False))
        self._add_checkbox(dciv, "Plot Result", "-MEAS_DCIV_PLOT-", default=True)
        self._add_checkbox(dciv, "Include HRS/LRS Probe Between Sweeps", "-MEAS_INCLUDE_READ_PROBE-", default=True)
        ttk.Button(dciv, text="Run Single DCIV Measurement", command=self._queue_single_dciv).pack(fill="x", pady=(6, 0))

        pulse = self._add_section(content, "Pulsed Measurement Parameters")
        self._add_entry_row(pulse, "Pulse Path", "-MEAS_PULSE_PATH-", DEFAULT_PULSE_PATH, width=40, browse="file", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        self._add_entry_row(pulse, "Compliance (A)", "-MEAS_PULSE_COMP-", "0.01", width=18)
        self._add_entry_row(pulse, "Pulse Width (s)", "-MEAS_PULSE_WIDTH-", "0.001", width=18)
        self._add_checkbox(pulse, "Use Current Autorange", "-SET_SINGLE_PULSE_AUTORANGE-", default=DEFAULT_RUNTIME_SETTINGS["single_pulse"]["current_autorange"], command=lambda: self._persist_runtime_settings(notify=False))
        self._add_checkbox(pulse, "Plot Result", "-MEAS_PULSE_PLOT-", default=True)
        ttk.Button(pulse, text="Run Single Pulsed Measurement", command=self._queue_single_pulse).pack(fill="x", pady=(6, 0))

    def _create_protocol_tab(self):
        frame = ttk.Frame(self.notebook, padding=8)
        frame.columnconfigure(0, weight=2)
        frame.columnconfigure(1, weight=3)
        frame.rowconfigure(0, weight=1)
        self.notebook.add(frame, text="Protocol Builder")

        left = ttk.Frame(frame)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = ttk.Frame(frame)
        right.grid(row=0, column=1, sticky="nsew")

        intro = self._add_section(left, "Protocol Builder")
        ttk.Label(intro, text="Build complex measurement sequences with full parameter control.", wraplength=260, justify="left").pack(anchor="w")
        ttk.Button(intro, text="Advanced Protocol", command=self._open_testmaker, style="Accent.TButton").pack(fill="x", pady=(8, 2))
        ttk.Button(intro, text="Add Step", command=self._add_protocol_step).pack(fill="x", pady=(8, 2))
        ttk.Button(intro, text="Edit Selected", command=self._edit_selected_protocol_step).pack(fill="x", pady=2)
        ttk.Button(intro, text="Remove Selected", command=self._remove_protocol_step).pack(fill="x", pady=2)
        ttk.Button(intro, text="Clear All", command=self._clear_protocol).pack(fill="x", pady=2)
        ttk.Separator(intro, orient="horizontal").pack(fill="x", pady=8)
        ttk.Button(intro, text="Save Protocol", command=self._save_protocol).pack(fill="x", pady=2)
        ttk.Button(intro, text="Load Protocol", command=self._load_protocol).pack(fill="x", pady=2)

        list_section = self._add_section(right, "Protocol Steps")
        list_frame = ttk.Frame(list_section)
        list_frame.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.protocol_listbox = tk.Listbox(
            list_frame,
            height=16,
            activestyle="dotbox",
            exportselection=False,
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            selectbackground=ACCENT_COLOR,
            selectforeground=TEXT_COLOR,
            highlightthickness=0,
            relief="flat",
        )
        self.protocol_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.protocol_listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.configure(command=self.protocol_listbox.yview)
        self.protocol_listbox.bind("<Double-Button-1>", lambda _event: self._edit_selected_protocol_step())
        self.protocol_listbox.bind("<Button-3>", self._show_protocol_context_menu)
        ttk.Label(list_section, text="Double-click a step to edit it. Right-click for edit/remove shortcuts.").pack(anchor="w", pady=(6, 0))
        ttk.Button(list_section, text="Run Protocol on Current Target", command=self._queue_run_protocol_single).pack(fill="x", pady=(8, 0))

        self.protocol_menu = tk.Menu(self.master, tearoff=False)
        self.protocol_menu.add_command(label="Edit Step", command=self._edit_selected_protocol_step)
        self.protocol_menu.add_command(label="Remove Step", command=self._remove_protocol_step)
        self.protocol_menu.configure(bg=PANEL_COLOR, fg=TEXT_COLOR, activebackground=ACCENT_COLOR, activeforeground=TEXT_COLOR)

    def _create_log_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Log Output", padding=8)
        panel.pack(fill="both", expand=True)
        status_row = ttk.Frame(panel)
        status_row.pack(fill="x", pady=(0, 6))
        ttk.Label(status_row, text="Task Status:").pack(side="left")
        ttk.Label(status_row, textvariable=self.task_status).pack(side="left", padx=(6, 0))
        tk.Button(
            status_row,
            text="ABORT",
            command=self._request_abort,
            bg="#991b1b",
            fg="white",
            activebackground="#b91c1c",
            activeforeground="white",
            relief="flat",
            padx=12,
            pady=4,
        ).pack(side="right")
        self.log_text = scrolledtext.ScrolledText(
            panel,
            wrap="word",
            height=40,
            state="disabled",
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            relief="flat",
        )
        self.log_text.pack(fill="both", expand=True)

    def _create_monitor_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Setup Monitor", padding=8)
        panel.pack(fill="x", pady=(0, 8))
        ttk.Label(panel, textvariable=self.monitor_detected_text, wraplength=360, justify="left").pack(anchor="w")
        ttk.Label(panel, textvariable=self.monitor_channels_text, wraplength=360, justify="left").pack(anchor="w", pady=(4, 0))
        ttk.Label(panel, textvariable=self.monitor_route_text, wraplength=360, justify="left").pack(anchor="w", pady=(4, 8))
        self.monitor_canvas = tk.Canvas(
            panel,
            width=360,
            height=240,
            bg=PANEL_COLOR,
            highlightthickness=1,
            highlightbackground=SURFACE_COLOR,
            relief="flat",
        )
        self.monitor_canvas.pack(fill="x", expand=True)
        self._draw_setup_diagram([], "Open")

    def _create_live_plot_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Live Plot", padding=8)
        panel.pack(fill="both", expand=False, pady=(0, 8))

        self.live_plot_figure = Figure(figsize=(4.0, 2.6), dpi=100, facecolor=PANEL_COLOR)
        self.live_plot_axis = self.live_plot_figure.add_subplot(111)
        self.live_plot_axis.set_facecolor(PANEL_COLOR)
        self.live_plot_axis.set_title(self.live_plot_title, color=TEXT_COLOR, fontsize=10)
        self.live_plot_axis.set_xlabel("Voltage (V)", color=TEXT_COLOR)
        self.live_plot_axis.set_ylabel("|Current| (A)", color=TEXT_COLOR)
        self.live_plot_axis.set_yscale("log")
        self.live_plot_axis.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in self.live_plot_axis.spines.values():
            spine.set_color(SURFACE_COLOR)
        self.live_plot_axis.grid(True, color=SURFACE_COLOR, alpha=0.5)
        (self.live_plot_line,) = self.live_plot_axis.plot([], [], color=SUCCESS_COLOR, linewidth=1.4)

        self.live_plot_canvas = FigureCanvasTkAgg(self.live_plot_figure, master=panel)
        self.live_plot_canvas.draw_idle()
        self.live_plot_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _schedule_monitor_refresh(self):
        self._refresh_setup_monitor()
        try:
            self._monitor_after_id = self.master.after(1000, self._schedule_monitor_refresh)
        except RuntimeError:
            self._monitor_after_id = None

    def on_close(self):
        if self._monitor_after_id is not None:
            try:
                self.master.after_cancel(self._monitor_after_id)
            except Exception:
                pass
            self._monitor_after_id = None
        if self.testmaker_app is not None:
            try:
                self.testmaker_app.root.destroy()
            except Exception:
                pass
            self.testmaker_app = None
        self._persist_runtime_settings(notify=False)
        self._stop_camera_thread()
        try:
            self._disconnect_all()
        except Exception:
            pass
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self.master.destroy()

    def _request_abort(self):
        if not (self.worker_thread and self.worker_thread.is_alive()):
            self._log("No background task is currently running.\n")
            return
        if self._is_abort_requested():
            self._log("Abort is already in progress.\n")
            return
        self.abort_event.set()
        self._set_task_status("Abort requested...")
        self._log("Abort requested. Worker will stop hardware safely.\n")

    def _refresh_setup_monitor(self):
        detected_parts = []
        if self.discovered_instruments.get("keithley"):
            detected_parts.append(f"Keithley: {self.discovered_instruments['keithley']}")
        if self.discovered_instruments.get("keysight"):
            detected_parts.append(f"Keysight: {self.discovered_instruments['keysight']}")
        if self.discovered_instruments.get("switch"):
            detected_parts.append(f"707B: {self.discovered_instruments['switch']}")
        if not detected_parts:
            detected_parts.append("none")
        self.monitor_detected_text.set("Detected equipment: " + " | ".join(detected_parts))

        channels, raw_channels = self._get_switch_channels()
        if isinstance(raw_channels, str) and raw_channels.startswith("error:"):
            self.monitor_channels_text.set(f"707B closed channels: {raw_channels}")
        elif channels:
            self.monitor_channels_text.set("707B closed channels: " + ", ".join(channels))
        else:
            self.monitor_channels_text.set("707B closed channels: none")

        route_name = self._identify_route(channels)
        active_label = self.active_smu_label.title() if self.active_smu_label else "None"
        self.monitor_route_text.set(f"Active route: {route_name} | Manual selection: {active_label}")
        self._draw_setup_diagram(channels, route_name)

    def _draw_setup_diagram(self, channels, route_name):
        canvas = getattr(self, "monitor_canvas", None)
        if canvas is None or not canvas.winfo_exists():
            return

        canvas.delete("all")

        detected = set(self.discovered_instruments.keys())
        active_route = route_name.lower()
        equipment_online = self.is_equipment_connected

        def box(x1, y1, x2, y2, label, fill, outline=None):
            outline = outline or fill
            canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=2)
            canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=label, fill=TEXT_COLOR, font=("Segoe UI", 9, "bold"))

        def line(x1, y1, x2, y2, color, width=3, dash=None):
            canvas.create_line(x1, y1, x2, y2, fill=color, width=width, dash=dash)

        inactive = "#4b5563"
        detected_color = "#2563eb"
        active_color = "#16a34a"
        warn_color = "#f59e0b"

        pc_fill = detected_color if equipment_online else inactive
        switch_fill = detected_color if "switch" in detected else inactive
        keithley_fill = active_color if active_route == "keithley" else (detected_color if "keithley" in detected else inactive)
        keysight_fill = active_color if active_route == "keysight" else (detected_color if "keysight" in detected else inactive)
        dut_fill = active_color if channels else inactive
        stage_fill = detected_color if equipment_online and getattr(self.vipsa, "stage", None) else inactive
        zaber_fill = detected_color if equipment_online and getattr(self.vipsa, "zaber_x", None) else inactive
        lights_fill = detected_color if equipment_online and getattr(self.vipsa, "top_light", None) else inactive

        box(20, 90, 90, 150, "PC", pc_fill)
        box(125, 90, 220, 150, "707B", switch_fill)
        box(255, 90, 340, 150, "DUT", dut_fill)
        box(20, 20, 110, 60, "Keithley", keithley_fill)
        box(20, 180, 110, 220, "Keysight", keysight_fill)
        box(125, 20, 220, 60, "Lights", lights_fill)
        box(125, 180, 220, 220, "Stage", stage_fill)
        box(255, 180, 340, 220, "Zaber", zaber_fill)

        line(90, 120, 125, 120, active_color if equipment_online else inactive)
        line(110, 40, 125, 90, active_color if active_route == "keithley" else inactive, dash=(4, 3) if active_route != "keithley" else None)
        line(110, 200, 125, 150, active_color if active_route == "keysight" else inactive, dash=(4, 3) if active_route != "keysight" else None)
        line(220, 120, 255, 120, active_color if channels else inactive)
        line(172, 60, 172, 90, detected_color if lights_fill != inactive else inactive, dash=(4, 3) if lights_fill == inactive else None)
        line(172, 150, 172, 180, warn_color if self._is_abort_requested() else (detected_color if stage_fill != inactive else inactive),
             dash=(4, 3) if stage_fill == inactive else None)
        line(297, 150, 297, 180, detected_color if zaber_fill != inactive else inactive, dash=(4, 3) if zaber_fill == inactive else None)

        canvas.create_text(
            180,
            232,
            text=f"Closed channels: {', '.join(channels) if channels else 'none'}",
            fill=MUTED_TEXT,
            font=("Segoe UI", 9),
        )

    def _reset_live_plot(self, title="Live Sweep"):
        self.live_plot_title = title
        self.live_plot_points = []
        axis = getattr(self, "live_plot_axis", None)
        line = getattr(self, "live_plot_line", None)
        canvas = getattr(self, "live_plot_canvas", None)
        if axis is None or line is None or canvas is None:
            return
        axis.clear()
        axis.set_facecolor(PANEL_COLOR)
        axis.set_title(self.live_plot_title, color=TEXT_COLOR, fontsize=10)
        axis.set_xlabel("Voltage (V)", color=TEXT_COLOR)
        axis.set_ylabel("|Current| (A)", color=TEXT_COLOR)
        axis.set_yscale("log")
        axis.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in axis.spines.values():
            spine.set_color(SURFACE_COLOR)
        axis.grid(True, color=SURFACE_COLOR, alpha=0.5)
        (self.live_plot_line,) = axis.plot([], [], color=SUCCESS_COLOR, linewidth=1.4)
        canvas.draw_idle()

    def _live_plot_callback(self, chunk, label=None):
        if chunk is None:
            return

        try:
            rows = list(chunk.tolist()) if hasattr(chunk, "tolist") else list(chunk)
        except Exception:
            rows = list(chunk)
        if not rows:
            return

        def update():
            if label and label != self.live_plot_title:
                self._reset_live_plot(label)

            for row in rows:
                if len(row) < 3:
                    continue
                voltage = float(row[1])
                current = abs(float(row[2]))
                self.live_plot_points.append((voltage, max(current, 1e-15)))

            if not self.live_plot_points:
                return

            xs = [point[0] for point in self.live_plot_points]
            ys = [point[1] for point in self.live_plot_points]
            self.live_plot_line.set_data(xs, ys)

            xmin = min(xs)
            xmax = max(xs)
            if xmin == xmax:
                xmin -= 0.1
                xmax += 0.1
            ymin = min(ys)
            ymax = max(ys)
            if ymin == ymax:
                ymin *= 0.5
                ymax *= 2.0

            self.live_plot_axis.set_xlim(xmin, xmax)
            self.live_plot_axis.set_ylim(max(ymin * 0.8, 1e-15), max(ymax * 1.2, 1e-14))
            self.live_plot_canvas.draw_idle()

        try:
            self.master.after(0, update)
        except RuntimeError:
            pass

    def _require_connection(self):
        if not self.is_equipment_connected:
            self._log("Error: Equipment not connected.\n")
            return False
        return True

    def _queue_measure_resistance(self):
        self._run_in_background("Resistance check", self._run_measure_resistance, self._collect_values())

    def _queue_single_dciv(self):
        self._run_in_background("Single DCIV measurement", self._run_single_dciv_meas, self._collect_values())

    def _queue_single_pulse(self):
        self._run_in_background("Single pulse measurement", self._run_single_pulse_meas, self._collect_values())

    def _queue_grid_measurement(self, measurement_type):
        self._run_in_background(f"Grid measurement ({measurement_type})", self._run_grid_measurement, self._collect_values(), measurement_type)

    def _queue_run_protocol_single(self):
        protocol = self.protocol_builder.protocol_list_configs
        if not protocol:
            messagebox.showerror("Protocol Empty", "Please add at least one step to the protocol.")
            self._log("Error: Protocol is empty.\n")
            return
        self._run_in_background("Single-device protocol run", self._run_protocol_single, self._collect_values())

    def _queue_quick_align(self):
        self._run_in_background("Quick align", self._quick_align)

    def _queue_quick_approach(self):
        self._run_in_background("Quick approach", self._quick_approach)

    def _set_zero(self):
        if getattr(self.vipsa, "stage", None):
            self.vipsa.stage.set_zero()
        else:
            self._log("Arduino stage not connected.\n")

    def _go_zero(self):
        if getattr(self.vipsa, "stage", None):
            self.vipsa.stage.go_to_zero()
        else:
            self._log("Arduino stage not connected.\n")

    def _start_camera_thread(self):
        if self.camera_thread and self.camera_thread.is_alive():
            self._log("Camera thread already running.\n")
            return
        self.camera_running = True
        self._cap = None
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        self._log("Camera thread started.\n")

    def _stop_camera_thread(self):
        self.camera_running = False
        if self._cap is not None:
            try:
                if self._cap.isOpened():
                    self._cap.release()
            except Exception:
                pass
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        self.camera_thread = None
        self._cap = None
        self.master.after(0, self._clear_camera_feeds)

    def _clear_camera_feeds(self):
        for label in self.camera_labels:
            if label.winfo_exists():
                label.configure(image="", text="Camera stopped", fg="white")
                label.image = None

    def _camera_loop(self):
        self._log("Camera loop starting...\n")
        try:
            cap = cv2.VideoCapture(0)
            self._cap = cap
            if not cap.isOpened():
                self._log("Error: Could not open camera.\n")
                self.camera_running = False
                return
            while self.camera_running:
                ret, frame = cap.read()
                if not ret:
                    if not self.camera_running:
                        break
                    self._log("Error: Could not read frame.\n")
                    time.sleep(0.1)
                    continue
                try:
                    processed = overlay(frame)
                    rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                    self.master.after(0, self._update_camera_feed, rgb_frame)
                except Exception as exc:
                    self._log(f"Error processing frame: {exc}\n")
                time.sleep(0.05)
        finally:
            if self._cap is not None:
                try:
                    if self._cap.isOpened():
                        self._cap.release()
                except Exception:
                    pass
            self._cap = None
            self.camera_running = False
            self._log("Camera loop finished.\n")

    def _update_camera_feed(self, rgb_frame):
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame).resize(CAMERA_SIZE))
        self._camera_photo = photo
        for label in self.camera_labels:
            if label.winfo_exists():
                label.configure(image=photo, text="")
                label.image = photo

    def _connect_all(self):
        selected_smu = self._get_value("-ACTIVE_SMU-", "Keithley2450")
        self._set_status("Connecting...", ACCENT_COLOR)
        try:
            self._log("Detecting USB/VISA equipment...\n")
            self._connect_real_equipment(selected_smu)
            self._log("All equipment connected successfully.\n")
        except Exception as exc:
            self._reset_connection_state()
            self._set_status("Error", ERROR_COLOR)
            self._log(f"Error connecting equipment: {exc}\n")
            self._refresh_setup_monitor()

    def _disconnect_all(self, log_message=True):
        current_smus = list(self.smus.values())
        current_stage = getattr(self.vipsa, "stage", None)
        current_zaber = getattr(self.vipsa, "Zaber", None)
        current_light = getattr(self.vipsa, "top_light", None)
        current_switch = self.switch
        self._cleanup_equipment_objects(
            smus=current_smus,
            stage_obj=current_stage,
            zaber_obj=current_zaber,
            top_light=current_light,
            switch_obj=current_switch,
        )
        self._reset_connection_state()
        self._set_status("Disconnected", ERROR_COLOR)
        if log_message:
            self._log("All equipment disconnected.\n")
        self._refresh_setup_monitor()

    def _switch_instrument(self):
        if not self._require_connection():
            return
        try:
            self._set_manual_active_smu(self._get_value("-ACTIVE_SMU-", "Keithley2450"), force_reconnect=True)
        except Exception as exc:
            self._log(f"Error switching instrument: {exc}\n")

    def _handle_arduino_move(self, axis, direction):
        if not self._require_connection() or not getattr(self.vipsa, "stage", None):
            return
        try:
            steps = float(self._get_value(f"-{axis}_STEPS-")) * direction
            self._log(f"Moving Arduino {axis} by {steps} steps...\n")
            if axis == "X":
                self.vipsa.stage.move_x_by(steps)
            elif axis == "Y":
                self.vipsa.stage.move_y_by(steps)
            else:
                self.vipsa.stage.move_z_by(steps)
        except ValueError:
            self._log(f"Error: Invalid {axis} step value.\n")
        except Exception as exc:
            self._log(f"Error moving Arduino {axis}: {exc}\n")

    def _handle_zaber_move(self, axis, direction):
        if not self._require_connection():
            return
        zaber_x = getattr(self.vipsa, "zaber_x", None)
        zaber_y = getattr(self.vipsa, "zaber_y", None)
        if zaber_x is None or zaber_y is None:
            self._log("Error: Zaber stages not connected.\n")
            return
        try:
            distance = float(self._get_value(f"-ZABER_{axis}_DIST-")) * direction
            self._log(f"Moving Zaber {axis} by {distance}...\n")
            if axis == "X":
                zaber_x.move_relative(distance)
            else:
                zaber_y.move_relative(distance)
        except ValueError:
            self._log(f"Error: Invalid Zaber {axis} distance.\n")
        except Exception as exc:
            self._log(f"Error moving Zaber {axis}: {exc}\n")

    def _control_lights(self, state):
        if not self._require_connection() or not getattr(self.vipsa, "top_light", None):
            self._log("Error: Lights not connected.\n")
            return
        try:
            self.vipsa.top_light.control_lights(state)
            self._log(f"Lights turned {state}.\n")
        except Exception as exc:
            self._log(f"Error controlling lights: {exc}\n")

    def _create_grid_csv(self, values):
        try:
            x_dist = float(values["-X_DIST-"])
            y_dist = float(values["-Y_DIST-"])
            rows = int(values["-ROWS-"])
            cols = int(values["-COLS-"])
            save_path = values["-GRID_PATH-"]
        except ValueError:
            self._log("Error: Invalid numeric input for grid parameters.\n")
            return
        if not save_path:
            self._log("Error: Please select a Grid Save Path.\n")
            return
        try:
            if getattr(self.vipsa, "zaber_x", None) is None or getattr(self.vipsa, "zaber_y", None) is None:
                port = self._get_value("-ZABER_PORT-", "COM4")
                self._log(f"Connecting Zaber on {port} for grid creation...\n")
                self.vipsa.Zaber = Zaber(port)
                self.vipsa.zaber_x, self.vipsa.zaber_y = self.vipsa.Zaber.get_devices()
            start_x = self._get_axis_position(self.vipsa.zaber_x)
            start_y = self._get_axis_position(self.vipsa.zaber_y)
            grid_points = []
            device_no = 1
            for row in range(rows):
                for col in range(cols):
                    grid_points.append([device_no, start_x + col * x_dist, start_y + row * y_dist])
                    device_no += 1
            os.makedirs(save_path, exist_ok=True)
            filepath = os.path.join(save_path, "grid.csv")
            with open(filepath, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Device", "X", "Y"])
                writer.writerows(grid_points)
            self._set_value("-GRID_MEAS_PATH-", filepath)
            self._log(f"Grid CSV created successfully at {filepath}\n")
        except Exception as exc:
            self._log(f"Error creating grid CSV: {exc}\n")

    def _quick_align(self):
        if self._require_connection():
            try:
                runtime_settings = self._build_runtime_settings()
                align_settings = runtime_settings["alignment"]
                self._log("Running quick align...\n")
                self.vipsa.correct_course(zaber_corr=align_settings["zaber_corr"], recheck=align_settings["recheck"])
            except Exception as exc:
                self._log(f"Encountered an issue during align: {exc}\n")

    def _quick_approach(self):
        if self._require_connection():
            try:
                runtime_settings = self._build_runtime_settings()
                self._set_manual_active_smu(self._get_value("-ACTIVE_SMU-", "Keithley2450"))
                self._log("Performing approach...\n")
                self.vipsa.detect_contact_and_move_z(
                    SMU=self.vipsa.SMU,
                    abort_requested=self._is_abort_requested,
                    **runtime_settings["quick_approach"],
                )
            except Exception as exc:
                self._log(f"Error during approach: {exc}\n")

    def _run_measure_resistance(self, values):
        if not self._require_connection():
            return
        try:
            self._set_manual_active_smu(self._get_value("-ACTIVE_SMU-", "Keithley2450"))
            voltage = float(values["-MEAS_RES_VOLTAGE-"])
            current = self.vipsa.SMU.get_contact_current(voltage)
            if current:
                self._log(f"Measured current: {current:.6e} A | Estimated resistance: {voltage / current:.6e} Ohm\n")
            else:
                self._log("Measured current is zero; resistance cannot be estimated.\n")
        except Exception as exc:
            self._log(f"Error during resistance check: {exc}\n")

    def _run_single_dciv_meas(self, values):
        if not self._require_connection():
            return
        try:
            runtime_settings = self._build_runtime_settings()
            align_settings = runtime_settings["alignment"]
            contact_settings = runtime_settings["single_dciv"]
            active_smu = self._set_manual_active_smu(self._get_value("-ACTIVE_SMU-", "Keithley2450"))
            sweep_path = values["-MEAS_SWEEP_PATH-"]
            if not self._require_file(sweep_path, "Sweep"):
                return
            measured, height, saved = self.vipsa.run_single_DCIV(
                sample_no=values["-MEAS_SAMPLE_ID-"],
                device_no=values["-MEAS_DEVICE_ID-"],
                pos_compl=float(values["-MEAS_POS_COMP-"]),
                neg_compl=float(values["-MEAS_NEG_COMP-"]),
                sweep_delay=float(values["-MEAS_SWEEP_DELAY-"]) if values["-MEAS_SWEEP_DELAY-"] else None,
                plot=False,
                align=values["-MEAS_ALIGN-"],
                approach=values["-MEAS_APPROACH-"],
                zaber_corr=align_settings["zaber_corr"],
                corr_recheck=align_settings["recheck"],
                save_directory=values["-MEAS_SAVE_FOLDER-"],
                sweep_path=sweep_path,
                wait_time=float(values["-WAIT_TIME-"]) if values["-WAIT_TIME-"] else 0.0,
                compliance_pf=float(values["-MEAS_POS_COMP-"]),
                compliance_pb=float(values["-MEAS_POS_COMP-"]),
                compliance_nf=float(values["-MEAS_NEG_COMP-"]),
                compliance_nb=float(values["-MEAS_NEG_COMP-"]),
                step_size=contact_settings["step_size"],
                test_voltage=contact_settings["test_voltage"],
                lower_threshold=contact_settings["lower_threshold"],
                upper_threshold=contact_settings["upper_threshold"],
                max_attempts=contact_settings["max_attempts"],
                delay=contact_settings["delay"],
                current_autorange=contact_settings["current_autorange"],
                use_4way_split=values["-MEAS_USE_4WAY_SPLIT-"],
                include_read_probe=values["-MEAS_INCLUDE_READ_PROBE-"],
                SMU=active_smu,
                stage=None,
                Zaber_x=None,
                Zaber_y=None,
                top_light=None,
                abort_requested=self._is_abort_requested,
                progress_callback=self._live_plot_callback,
            )
            if measured:
                self._log(f"Single DCIV finished. Height={height}, saved to {saved}\n")
                if values["-MEAS_DCIV_PLOT-"]:
                    self._schedule_saved_plot(
                        "Sweep",
                        saved,
                        values["-MEAS_SAMPLE_ID-"],
                        values["-MEAS_DEVICE_ID-"],
                    )
            else:
                self._log("Single DCIV failed or contact was not established.\n")
        except Exception as exc:
            self._log(f"Error during single DCIV measurement: {exc}\n")

    def _run_single_pulse_meas(self, values):
        if not self._require_connection():
            return
        try:
            runtime_settings = self._build_runtime_settings()
            align_settings = runtime_settings["alignment"]
            contact_settings = runtime_settings["single_pulse"]
            active_smu = self._set_manual_active_smu(self._get_value("-ACTIVE_SMU-", "Keithley2450"))
            pulse_path = values["-MEAS_PULSE_PATH-"]
            if not self._require_file(pulse_path, "Pulse"):
                return
            measured, height, saved = self.vipsa.run_single_pulse(
                sample_no=values["-MEAS_SAMPLE_ID-"],
                device_no=values["-MEAS_DEVICE_ID-"],
                compliance=float(values["-MEAS_PULSE_COMP-"]),
                pulse_width=float(values["-MEAS_PULSE_WIDTH-"]) if values["-MEAS_PULSE_WIDTH-"] else None,
                plot=False,
                align=values["-MEAS_ALIGN-"],
                approach=values["-MEAS_APPROACH-"],
                zaber_corr=align_settings["zaber_corr"],
                corr_recheck=align_settings["recheck"],
                step_size=contact_settings["step_size"],
                test_voltage=contact_settings["test_voltage"],
                lower_threshold=contact_settings["lower_threshold"],
                upper_threshold=contact_settings["upper_threshold"],
                max_attempts=contact_settings["max_attempts"],
                delay=contact_settings["delay"],
                current_autorange=contact_settings["current_autorange"],
                save_directory=values["-MEAS_SAVE_FOLDER-"],
                pulse_path=pulse_path,
                SMU=active_smu,
                stage=None,
                Zaber_x=None,
                Zaber_y=None,
                top_light=None,
                abort_requested=self._is_abort_requested,
            )
            if measured:
                self._log(f"Single pulse finished. Height={height}, saved to {saved}\n")
                if values["-MEAS_PULSE_PLOT-"]:
                    self._schedule_saved_plot(
                        "Pulse",
                        saved,
                        values["-MEAS_SAMPLE_ID-"],
                        values["-MEAS_DEVICE_ID-"],
                    )
            else:
                self._log("Single pulse failed or contact was not established.\n")
        except Exception as exc:
            self._log(f"Error during single pulse measurement: {exc}\n")

    def _iter_grid_devices(self, grid_path, startpoint, skip, randomize_order):
        devices = pd.read_csv(grid_path)[["Device", "X", "Y"]].values.tolist()[max(startpoint - 1, 0) :]
        if skip > 1:
            devices = devices[::skip]
        if randomize_order:
            random.shuffle(devices)
        return devices

    def _run_grid_measurement(self, values, measurement_type="DCIV"):
        if not self._require_connection():
            return
        try:
            runtime_settings = self._build_runtime_settings()
            align_settings = runtime_settings["alignment"]
            contact_settings = runtime_settings["grid"]
            active_smu = self._set_manual_active_smu(self._get_value("-ACTIVE_SMU-", "Keithley2450"))
            grid_path = values["-GRID_MEAS_PATH-"]
            if not self._require_file(grid_path, "Grid"):
                return
            sample_id = values["-GRID_SAMPLE_ID-"]
            startpoint = int(values["-GRID_STARTPOINT-"])
            skip = max(1, int(values["-GRID_SKIP-"]))
            randomize_order = bool(values["-GRID_RANDOMIZE-"])
            save_dir = values["-MEAS_SAVE_FOLDER-"]
            self._log(f"Starting grid measurement ({measurement_type})...\n")
            if measurement_type == "DCIV":
                sweep_path = values["-GRID_SWEEP_PATH-"]
                if not self._require_file(sweep_path, "Grid DCIV sweep"):
                    return
                for dev_id, x, y in self._iter_grid_devices(grid_path, startpoint, skip, randomize_order):
                    if self._is_abort_requested():
                        self._log("Abort acknowledged during grid DCIV run.\n")
                        break
                    self._log(f"Running DCIV on grid device {int(dev_id)}...\n")
                    self.vipsa.zaber_x.move_absolute(x)
                    self.vipsa.zaber_y.move_absolute(y)
                    measured, height, saved = self.vipsa.run_single_DCIV(
                        sample_no=sample_id,
                        device_no=int(dev_id),
                        pos_compl=float(values["-GRID_POS_COMP-"]),
                        neg_compl=float(values["-GRID_NEG_COMP-"]),
                        sweep_delay=float(values["-GRID_SWEEP_DELAY-"]) if values["-GRID_SWEEP_DELAY-"] else None,
                        plot=False,
                        align=True,
                        approach=True,
                        zaber_corr=align_settings["zaber_corr"],
                        corr_recheck=align_settings["recheck"],
                        step_size=contact_settings["step_size"],
                        test_voltage=contact_settings["test_voltage"],
                        lower_threshold=contact_settings["lower_threshold"],
                        upper_threshold=contact_settings["upper_threshold"],
                        max_attempts=contact_settings["max_attempts"],
                        delay=contact_settings["delay"],
                        current_autorange=contact_settings["current_autorange"],
                        save_directory=save_dir,
                        sweep_path=sweep_path,
                        compliance_pf=float(values["-GRID_POS_COMP-"]),
                        compliance_pb=float(values["-GRID_POS_COMP-"]),
                        compliance_nf=float(values["-GRID_NEG_COMP-"]),
                        compliance_nb=float(values["-GRID_NEG_COMP-"]),
                        SMU=active_smu,
                        stage=None,
                        Zaber_x=None,
                        Zaber_y=None,
                        top_light=None,
                        use_4way_split=values["-GRID_USE_4WAY_SPLIT-"],
                        include_read_probe=values["-GRID_INCLUDE_READ_PROBE-"],
                        abort_requested=self._is_abort_requested,
                        progress_callback=self._live_plot_callback,
                    )
                    if measured:
                        self._log(f"Grid DCIV complete for device {int(dev_id)}. Height={height}, saved to {saved}\n")
                    else:
                        self._log(f"Grid DCIV failed for device {int(dev_id)}.\n")
            elif measurement_type == "PULSE":
                pulse_path = values["-GRID_PULSE_PATH-"]
                if not self._require_file(pulse_path, "Grid pulse"):
                    return
                for dev_id, x, y in self._iter_grid_devices(grid_path, startpoint, skip, randomize_order):
                    if self._is_abort_requested():
                        self._log("Abort acknowledged during grid pulse run.\n")
                        break
                    self._log(f"Running pulse on grid device {int(dev_id)}...\n")
                    self.vipsa.zaber_x.move_absolute(x)
                    self.vipsa.zaber_y.move_absolute(y)
                    self.vipsa.run_single_pulse(
                        sample_no=sample_id,
                        device_no=int(dev_id),
                        compliance=float(values["-GRID_PULSE_COMP-"]),
                        pulse_width=float(values["-GRID_PULSE_WIDTH-"]) if values["-GRID_PULSE_WIDTH-"] else None,
                        plot=False,
                        align=True,
                        approach=True,
                        zaber_corr=align_settings["zaber_corr"],
                        corr_recheck=align_settings["recheck"],
                        step_size=contact_settings["step_size"],
                        test_voltage=contact_settings["test_voltage"],
                        lower_threshold=contact_settings["lower_threshold"],
                        upper_threshold=contact_settings["upper_threshold"],
                        max_attempts=contact_settings["max_attempts"],
                        delay=contact_settings["delay"],
                        current_autorange=contact_settings["current_autorange"],
                        save_directory=save_dir,
                        pulse_path=pulse_path,
                        SMU=active_smu,
                        stage=None,
                        Zaber_x=None,
                        Zaber_y=None,
                        top_light=None,
                        abort_requested=self._is_abort_requested,
                    )
            else:
                protocol = self.protocol_builder.protocol_list_configs
                if not protocol:
                    self._log("Error: Protocol is empty. Cannot run grid protocol.\n")
                    return
                for dev_id, x, y in self._iter_grid_devices(grid_path, startpoint, skip, randomize_order):
                    if self._is_abort_requested():
                        self._log("Abort acknowledged during grid protocol run.\n")
                        break
                    self._log(f"Running protocol on grid device {int(dev_id)}...\n")
                    self.vipsa.zaber_x.move_absolute(x)
                    self.vipsa.zaber_y.move_absolute(y)
                    device_save_dir = os.path.join(save_dir, sample_id, f"device_{int(dev_id)}")
                    os.makedirs(device_save_dir, exist_ok=True)
                    protocol_smus = self._ensure_protocol_smus(protocol)
                    results = self.vipsa.run_protocol(
                        protocol,
                        sample_no=sample_id,
                        device_no=int(dev_id),
                        save_directory=device_save_dir,
                        SMU=protocol_smus,
                        stop_on_error=False,
                        abort_requested=self._is_abort_requested,
                        progress_callback=self._live_plot_callback,
                        runtime_settings=runtime_settings,
                        runtime_profile="grid",
                    )
                    self._log(f"Protocol results for device {int(dev_id)}: {results}\n")
            self._log(f"Grid measurement ({measurement_type}) finished.\n")
        except Exception as exc:
            self._log(f"Error during grid measurement: {exc}\n")

    def _selected_protocol_index(self):
        selection = self.protocol_listbox.curselection()
        return selection[0] if selection else None

    def _show_protocol_context_menu(self, event):
        if self.protocol_listbox.size() == 0:
            return
        try:
            index = self.protocol_listbox.nearest(event.y)
            self.protocol_listbox.selection_clear(0, tk.END)
            self.protocol_listbox.selection_set(index)
            self.protocol_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.protocol_menu.grab_release()

    def _refresh_protocol_display(self):
        self.protocol_listbox.delete(0, tk.END)
        for item in self.protocol_builder.get_protocol_display_list():
            self.protocol_listbox.insert(tk.END, item)

    def _add_protocol_step(self):
        result = self.protocol_builder.show_step_editor()
        if result:
            self._log(f"{result}\n")
            self._refresh_protocol_display()

    def _edit_selected_protocol_step(self):
        index = self._selected_protocol_index()
        if index is None:
            self._log("No protocol step selected for editing.\n")
            return
        result = self.protocol_builder.show_step_editor(initial_step=self.protocol_builder.protocol_list_configs[index], edit_index=index)
        if result:
            self._log(f"{result}\n")
            self._refresh_protocol_display()

    def _remove_protocol_step(self):
        index = self._selected_protocol_index()
        if index is None:
            self._log("No protocol step selected for removal.\n")
            return
        result = self.protocol_builder.remove_step(index)
        if result:
            self._log(f"{result}\n")
            self._refresh_protocol_display()

    def _clear_protocol(self):
        if messagebox.askyesno("Clear Protocol", "Clear all protocol steps?"):
            self.protocol_builder.clear_protocol()
            self._refresh_protocol_display()
            self._log("Protocol cleared.\n")

    def _save_protocol(self):
        filepath = filedialog.asksaveasfilename(parent=self.master, title="Save Protocol", defaultextension=".json", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if filepath and self.protocol_builder.export_protocol(filepath):
            self._log(f"Protocol saved to {filepath}\n")

    def _load_protocol(self):
        filepath = filedialog.askopenfilename(parent=self.master, title="Load Protocol", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if filepath and self.protocol_builder.import_protocol(filepath):
            self._refresh_protocol_display()
            self._log(f"Protocol loaded from {filepath}\n")

    def _run_protocol_single(self, values):
        if not self._require_connection():
            return
        protocol = self.protocol_builder.protocol_list_configs
        if not protocol:
            messagebox.showerror("Protocol Empty", "Please add at least one step to the protocol.")
            self._log("Error: Protocol is empty.\n")
            return
        sample_id = values.get("-MEAS_SAMPLE_ID-", "sample")
        device_id = values.get("-MEAS_DEVICE_ID-", "0")
        save_dir = values.get("-MEAS_SAVE_FOLDER-", DEFAULT_SAVE_DIRECTORY)
        try:
            device_no = int(device_id)
        except Exception:
            device_no = device_id
        device_save_dir = os.path.join(save_dir, sample_id, f"device_{device_no}")
        os.makedirs(device_save_dir, exist_ok=True)
        try:
            self._log(f"Running protocol on sample {sample_id}, device {device_no}...\n")
            protocol_smus = self._ensure_protocol_smus(protocol)
            results = self.vipsa.run_protocol(
                protocol,
                sample_no=sample_id,
                device_no=device_no,
                save_directory=device_save_dir,
                SMU=protocol_smus,
                abort_requested=self._is_abort_requested,
                progress_callback=self._live_plot_callback,
                runtime_settings=self._build_runtime_settings(),
                runtime_profile="single",
            )
            self._log(f"Protocol run complete. Results: {results}\n")
        except Exception as exc:
            self._log(f"Error while running protocol: {exc}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = VipsaGUI(root)
    root.mainloop()
