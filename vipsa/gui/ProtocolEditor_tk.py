import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


STEP_TEMPLATES = {
    "DCIV": {
        "display_name": "DC/IV Sweep",
        "params": {
            "sweep_path": "",
            "pos_compl": 0.001,
            "neg_compl": 0.01,
            "sweep_delay": 0.0001,
            "align": False,
            "approach": False,
            "smu_select": "Keithley2450",
            "use_4way_split": True,
            "include_read_probe": True,
            "read_probe_mode": "between_segments",
        },
    },
    "PULSE": {
        "display_name": "Pulsed Measurement",
        "params": {
            "pulse_path": "",
            "compliance": 0.01,
            "pulse_width": 0.001,
            "align": False,
            "approach": False,
            "smu_select": "KeysightB2901BL",
            "set_acquire_delay": 0.0005,
        },
    },
    "CV_CURRENT_PROBE": {
        "display_name": "Constant Voltage Current Probe",
        "params": {
            "voltage": 0.1,
            "duration": 1.0,
            "sample_interval": 0.1,
            "compliance": 0.001,
            "align": False,
            "approach": False,
            "smu_select": "Keithley2450",
            "current_autorange": False,
        },
    },
    "ALIGN": {
        "display_name": "Correct Course (Align)",
        "params": {
            "move": True,
            "zaber_corr": True,
            "recheck": True,
        },
    },
    "APPROACH": {
        "display_name": "Detect Contact & Approach",
        "params": {
            "step_size": 0.5,
            "test_voltage": 0.1,
            "lower_threshold": 1e-11,
            "upper_threshold": 5e-11,
            "max_attempts": 50,
            "delay": 1.0,
        },
    },
    "RESISTANCE": {
        "display_name": "Resistance Measurement",
        "params": {
            "sweep_path": "",
            "pos_compl": 0.001,
            "neg_compl": 0.01,
            "sweep_delay": 0.0001,
            "align": False,
            "approach": False,
            "smu_select": "Keithley2450",
        },
    },
    "DELAY": {
        "display_name": "Delay / Wait",
        "params": {
            "duration": 1.0,
            "note": "",
        },
    },
    "LOG_MESSAGE": {
        "display_name": "Log Message",
        "params": {
            "message": "Protocol checkpoint",
        },
    },
    "CUSTOM": {
        "display_name": "Custom Sequence",
        "params": {
            "description": "Define custom JSON parameters",
            "json_params": "{}",
        },
    },
}


class ProtocolStepEditorDialog(tk.Toplevel):
    def __init__(self, parent, initial_step=None):
        super().__init__(parent)
        self.parent = parent
        self.result = None
        self.step_type_var = tk.StringVar(value=(initial_step or {}).get("type", "DCIV"))
        self.param_vars = {}
        self.text_widgets = {}
        self.initial_step = initial_step or {}

        self.title("Protocol Step Editor")
        self.geometry("620x680")
        self.minsize(560, 540)
        self.transient(parent)
        self.grab_set()

        self._build_ui()
        self._rebuild_fields(self.initial_step)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _build_ui(self):
        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="Protocol Step Editor", font=("Arial", 14, "bold")).pack(anchor="w")
        ttk.Separator(container, orient="horizontal").pack(fill="x", pady=8)

        selector = ttk.Frame(container)
        selector.pack(fill="x", pady=(0, 8))
        ttk.Label(selector, text="Step Type", width=18).pack(side="left")
        combo = ttk.Combobox(selector, textvariable=self.step_type_var, values=tuple(STEP_TEMPLATES.keys()), state="readonly")
        combo.pack(side="left", fill="x", expand=True)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._rebuild_fields())

        self.canvas = tk.Canvas(container, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.form = ttk.Frame(self.canvas)
        self.form_window = self.canvas.create_window((0, 0), window=self.form, anchor="nw")
        self.form.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.form_window, width=event.width))

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=12, pady=12)
        ttk.Button(actions, text="Save Step", command=self._save).pack(side="right")
        ttk.Button(actions, text="Cancel", command=self._cancel).pack(side="right", padx=(0, 6))

    def _browse_file(self, key):
        path = filedialog.askopenfilename(parent=self)
        if path:
            self.param_vars[key].set(path)

    def _clear_fields(self):
        for child in self.form.winfo_children():
            child.destroy()
        self.param_vars.clear()
        self.text_widgets.clear()

    def _rebuild_fields(self, initial_step=None):
        self._clear_fields()
        template = STEP_TEMPLATES[self.step_type_var.get()]
        params = dict(template["params"])
        if initial_step:
            params.update(initial_step.get("params", {}))

        ttk.Label(self.form, text=f"Configure {template['display_name']}", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 6))
        for key, default in params.items():
            if isinstance(default, bool):
                var = tk.BooleanVar(value=default)
                self.param_vars[key] = var
                ttk.Checkbutton(self.form, text=key.replace("_", " ").title(), variable=var).pack(anchor="w", pady=2)
                continue

            if key in {"message", "description", "note", "json_params"}:
                ttk.Label(self.form, text=key.replace("_", " ").title()).pack(anchor="w", pady=(6, 0))
                text = tk.Text(self.form, height=6 if key == "json_params" else 3, wrap="word")
                text.pack(fill="x", expand=True)
                text.insert("1.0", str(default))
                self.text_widgets[key] = text
                continue

            row = ttk.Frame(self.form)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=key.replace("_", " ").title(), width=20).pack(side="left")
            if key == "smu_select":
                var = tk.StringVar(value=str(default))
                ttk.Combobox(row, textvariable=var, values=("Keithley2450", "KeysightB2901BL"), state="readonly").pack(side="left", fill="x", expand=True)
                self.param_vars[key] = var
            else:
                var = tk.StringVar(value=str(default))
                ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
                self.param_vars[key] = var
                if key.endswith("_path"):
                    ttk.Button(row, text="Browse", command=lambda k=key: self._browse_file(k)).pack(side="left", padx=(6, 0))

    def _extract_params(self):
        template = STEP_TEMPLATES[self.step_type_var.get()]
        params = {}
        for key, default in template["params"].items():
            if key in self.text_widgets:
                raw = self.text_widgets[key].get("1.0", tk.END).strip()
                params[key] = json.loads(raw or "{}") if key == "json_params" else raw
                continue
            value = self.param_vars[key].get()
            if isinstance(default, bool):
                params[key] = bool(value)
            elif isinstance(default, int) and not isinstance(default, bool):
                params[key] = int(value)
            elif isinstance(default, float):
                params[key] = float(value)
            else:
                params[key] = value
        return params

    def _validate(self, step_type, params):
        try:
            if step_type in {"DCIV", "RESISTANCE"}:
                return params["pos_compl"] > 0 and params["neg_compl"] > 0 and params["sweep_delay"] >= 0
            if step_type == "PULSE":
                return params["compliance"] > 0 and params["pulse_width"] > 0
            if step_type == "CV_CURRENT_PROBE":
                return (
                    params["compliance"] > 0
                    and params["duration"] >= 0
                    and params["sample_interval"] > 0
                )
            if step_type == "APPROACH":
                return params["step_size"] > 0 and params["test_voltage"] > 0 and params["max_attempts"] > 0
            if step_type == "DELAY":
                return params["duration"] >= 0
            return True
        except Exception:
            return False

    def _save(self):
        step_type = self.step_type_var.get()
        try:
            params = self._extract_params()
        except (ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Invalid Parameters", f"Could not parse step parameters:\n{exc}", parent=self)
            return
        if not self._validate(step_type, params):
            messagebox.showerror("Invalid Parameters", "Please check the step parameters and try again.", parent=self)
            return
        self.result = {"type": step_type, "params": params}
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class ProtocolBuilderTk:
    def __init__(self, parent_window, vipsa_instance, logger=None):
        self.parent_window = parent_window
        self.vipsa = vipsa_instance
        self.logger = logger
        self.protocol_list_configs = []

    def show_step_editor(self, initial_step=None, edit_index=None):
        dialog = ProtocolStepEditorDialog(self.parent_window, initial_step=initial_step)
        self.parent_window.wait_window(dialog)
        if not dialog.result:
            return None
        if edit_index is None:
            self.protocol_list_configs.append(dialog.result)
            return f"Added: {dialog.result['type']}"
        self.protocol_list_configs[edit_index] = dialog.result
        return f"Updated: {dialog.result['type']}"

    def get_protocol_display_list(self):
        display = []
        for index, step in enumerate(self.protocol_list_configs, start=1):
            step_type = step.get("type", "UNKNOWN")
            params = step.get("params", {})
            parts = [f"{index}. {step_type}"]
            if step_type in {"DCIV", "RESISTANCE"}:
                parts.append(f"[Pos Compl: {params.get('pos_compl')}]")
                parts.append(f"[SMU: {params.get('smu_select', 'Keithley2450')}]")
                if step_type == "DCIV" and not params.get("include_read_probe", True):
                    parts.append("[Read Probe: Off]")
            elif step_type == "PULSE":
                parts.append(f"[Compliance: {params.get('compliance')}]")
                parts.append(f"[Width: {params.get('pulse_width')}]")
                parts.append(f"[SMU: {params.get('smu_select', 'KeysightB2901BL')}]")
            elif step_type == "CV_CURRENT_PROBE":
                parts.append(f"[V: {params.get('voltage')} V]")
                parts.append(f"[T: {params.get('duration')} s]")
                parts.append(f"[dt: {params.get('sample_interval')} s]")
                parts.append(f"[SMU: {params.get('smu_select', 'Keithley2450')}]")
            elif step_type == "APPROACH":
                parts.append(f"[Threshold: {params.get('lower_threshold')}..{params.get('upper_threshold')}]")
            elif step_type == "DELAY":
                parts.append(f"[Duration: {params.get('duration')} s]")
            elif step_type == "LOG_MESSAGE":
                parts.append(f"[{params.get('message', '')}]")
            display.append(" ".join(parts))
        return display

    def export_protocol(self, filepath):
        try:
            return self.vipsa.save_protocol(filepath, self.protocol_list_configs)
        except Exception as exc:
            messagebox.showerror("Save Failed", f"Error saving protocol:\n{exc}", parent=self.parent_window)
            return False

    def import_protocol(self, filepath):
        try:
            protocol = self.vipsa.load_protocol(filepath)
            ok, message = self.vipsa.validate_protocol(protocol)
            if not ok:
                messagebox.showerror("Invalid Protocol", message, parent=self.parent_window)
                return False
            self.protocol_list_configs = protocol
            return True
        except Exception as exc:
            messagebox.showerror("Load Failed", f"Error loading protocol:\n{exc}", parent=self.parent_window)
            return False

    def clear_protocol(self):
        self.protocol_list_configs = []
        return True

    def remove_step(self, index):
        if 0 <= index < len(self.protocol_list_configs):
            removed = self.protocol_list_configs.pop(index)
            return f"Removed: {removed['type']}"
        return None
