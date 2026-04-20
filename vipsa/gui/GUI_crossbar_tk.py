from __future__ import annotations

import json
import os
import time
import tkinter as tk
import threading
from tkinter import filedialog, messagebox, scrolledtext, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from vipsa.analysis.Datahandling import Data_Handler
from vipsa.gui.ProtocolEditor_tk import ProtocolBuilderTk
from vipsa.workflows.Main_crossbar import Crossbar_Methods


DEFAULT_SAVE_DIRECTORY = "./crossbar_output"


class CrossbarTkApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ViPSA Crossbar Control")
        self.root.geometry("1500x920")
        self.root.minsize(1280, 820)

        self.crossbar = Crossbar_Methods()
        self.data_handler = Data_Handler()
        self.protocol_builder = ProtocolBuilderTk(self.root, self._protocol_file_adapter())

        self.vars = {}
        self.grid_rects = {}
        self.selection_state = {}
        self.selected_device_count = 0
        self.worker = None
        self.live_plot_title = "Live Sweep"
        self.live_plot_points = []

        self._build_ui()
        self._draw_grid()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _protocol_file_adapter(self):
        class Adapter:
            @staticmethod
            def save_protocol(filepath, protocol):
                with open(filepath, "w", encoding="utf-8") as handle:
                    json.dump(protocol, handle, indent=2)
                return True

            @staticmethod
            def load_protocol(filepath):
                with open(filepath, "r", encoding="utf-8") as handle:
                    return json.load(handle)

            @staticmethod
            def validate_protocol(protocol):
                if not isinstance(protocol, list):
                    return False, "Protocol must be a list."
                return True, "OK"

        return Adapter()

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
        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill="both", expand=True)
        left = ttk.Frame(main, padding=10)
        right = ttk.Frame(main, padding=10)
        main.add(left, weight=4)
        main.add(right, weight=2)

        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill="both", expand=True)

        self._create_grid_tab()
        self._create_manual_tab()
        self._create_protocol_tab()
        self._create_log_tab()

        self._create_live_plot_panel(right)
        actions = ttk.LabelFrame(right, text="Graph Actions", padding=10)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Button(actions, text="Save Current Graph", command=self._save_current_graph).pack(fill="x")
        ttk.Button(actions, text="Reset Plot", command=lambda: self._reset_live_plot("Live Sweep")).pack(fill="x", pady=(6, 0))

    def _create_grid_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Grid View")
        left = ttk.Frame(frame)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(frame)
        right.pack(side="left", fill="y", padx=(12, 0))

        ttk.Label(left, text="Multiplexer Grid View", font=("Arial", 12, "bold")).pack(anchor="w")
        ttk.Label(left, textvariable=self._string_var("SELECTED_COUNT", "Devices Selected: 0")).pack(anchor="w", pady=(4, 8))
        self.grid_canvas = tk.Canvas(left, width=640, height=640, bg="light gray", highlightthickness=1, highlightbackground="#999")
        self.grid_canvas.pack(fill="both", expand=False)
        self.grid_canvas.bind("<Button-1>", self._on_grid_click)

        ttk.Button(right, text="Select All", command=self._select_all).pack(fill="x", pady=2)
        ttk.Button(right, text="Deselect All", command=self._deselect_all).pack(fill="x", pady=2)
        ttk.Button(right, text="Select Half", command=self._select_half).pack(fill="x", pady=2)
        ttk.Button(right, text="Toggle Selection", command=self._toggle_selection).pack(fill="x", pady=2)
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)
        ttk.Radiobutton(right, text="Run DCIV", variable=self._string_var("GRID_ACTION", "DCIV"), value="DCIV").pack(anchor="w")
        ttk.Radiobutton(right, text="Run Pulse", variable=self._string_var("GRID_ACTION", "DCIV"), value="PULSE").pack(anchor="w")
        ttk.Radiobutton(right, text="Run Current Probe", variable=self._string_var("GRID_ACTION", "DCIV"), value="CV_CURRENT_PROBE").pack(anchor="w")
        ttk.Radiobutton(right, text="Run Protocol", variable=self._string_var("GRID_ACTION", "DCIV"), value="PROTOCOL").pack(anchor="w")
        ttk.Button(right, text="Measure Selected Devices", command=self.queue_measure_selected).pack(fill="x", pady=(12, 0))

    def _create_manual_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Manual Measurement")

        conn = ttk.LabelFrame(frame, text="Connections", padding=10)
        conn.pack(fill="x", pady=(0, 10))
        ttk.Button(conn, text="Connect Equipments", command=self._connect).grid(row=0, column=0, sticky="w")
        ttk.Button(conn, text="Disconnect Equipments", command=self._disconnect).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(conn, textvariable=self._string_var("STATUS", "Status: Disconnected")).grid(row=0, column=2, sticky="w", padx=(12, 0))
        self._add_entry(conn, "Channel 1 Pin", "CH1", "1", 1, 0)
        self._add_entry(conn, "Channel 2 Pin", "CH2", "1", 1, 2)
        ttk.Button(conn, text="Set Manual Channels", command=self._set_channels).grid(row=1, column=4, sticky="w", padx=(8, 0), pady=(6, 0))

        meta = ttk.LabelFrame(frame, text="Run Context", padding=10)
        meta.pack(fill="x", pady=(0, 10))
        self._add_entry(meta, "Sample ID", "SAMPLE", "crossbar", 0, 0)
        self._add_entry(meta, "Operator", "OPERATOR", "", 0, 2)
        ttk.Label(meta, text="Save Folder").grid(row=1, column=0, sticky="w", pady=(6, 0))
        folder_row = ttk.Frame(meta)
        folder_row.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(6, 0))
        folder_row.columnconfigure(0, weight=1)
        ttk.Entry(folder_row, textvariable=self._string_var("SAVE_FOLDER", DEFAULT_SAVE_DIRECTORY)).grid(row=0, column=0, sticky="ew")
        ttk.Button(folder_row, text="Browse", command=lambda: self._choose_directory("SAVE_FOLDER")).grid(row=0, column=1, padx=(6, 0))
        ttk.Label(meta, text="Notes").grid(row=2, column=0, sticky="nw", pady=(6, 0))
        self.notes_text = tk.Text(meta, height=4, wrap="word")
        self.notes_text.grid(row=2, column=1, columnspan=3, sticky="ew", pady=(6, 0))
        for col in range(4):
            meta.columnconfigure(col, weight=1 if col % 2 else 0)

        dciv = ttk.LabelFrame(frame, text="DCIV Sweep", padding=10)
        dciv.pack(fill="x", pady=(0, 10))
        ttk.Label(dciv, text="Sweep CSV").grid(row=0, column=0, sticky="w")
        path_row = ttk.Frame(dciv)
        path_row.grid(row=0, column=1, columnspan=5, sticky="ew")
        path_row.columnconfigure(0, weight=1)
        ttk.Entry(path_row, textvariable=self._string_var("SWEEP_PATH", "")).grid(row=0, column=0, sticky="ew")
        ttk.Button(path_row, text="Browse", command=lambda: self._choose_file("SWEEP_PATH")).grid(row=0, column=1, padx=(6, 0))
        self._add_entry(dciv, "Pos Comp (A)", "DCIV_POS", "0.001", 1, 0)
        self._add_entry(dciv, "Neg Comp (A)", "DCIV_NEG", "0.01", 1, 2)
        self._add_entry(dciv, "Sweep Delay", "DCIV_DELAY", "0.0001", 1, 4)
        self._add_entry(dciv, "Acq Delay", "DCIV_ACQ", "", 2, 0)
        ttk.Checkbutton(dciv, text="Use Current Autorange", variable=self._bool_var("DCIV_AUTORANGE", False)).grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(dciv, text="Use 4-way Split", variable=self._bool_var("DCIV_4WAY", True)).grid(row=2, column=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(dciv, text="Include Read Probe", variable=self._bool_var("DCIV_READ_PROBE", True)).grid(row=2, column=4, sticky="w", pady=(6, 0))
        ttk.Button(dciv, text="Run Single DCIV", command=self.queue_run_single_dciv).grid(row=3, column=0, sticky="w", pady=(8, 0))

        pulse = ttk.LabelFrame(frame, text="Pulse Measurement", padding=10)
        pulse.pack(fill="x")
        ttk.Label(pulse, text="Pulse CSV").grid(row=0, column=0, sticky="w")
        pulse_row = ttk.Frame(pulse)
        pulse_row.grid(row=0, column=1, columnspan=5, sticky="ew")
        pulse_row.columnconfigure(0, weight=1)
        ttk.Entry(pulse_row, textvariable=self._string_var("PULSE_PATH", "")).grid(row=0, column=0, sticky="ew")
        ttk.Button(pulse_row, text="Browse", command=lambda: self._choose_file("PULSE_PATH")).grid(row=0, column=1, padx=(6, 0))
        self._add_entry(pulse, "Compliance (A)", "PULSE_COMP", "0.01", 1, 0)
        self._add_entry(pulse, "Pulse Width", "PULSE_WIDTH", "", 1, 2)
        self._add_entry(pulse, "Acquire Delay", "PULSE_ACQ", "", 1, 4)
        ttk.Checkbutton(pulse, text="Use Current Autorange", variable=self._bool_var("PULSE_AUTORANGE", False)).grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Button(pulse, text="Run Single Pulse", command=self.queue_run_single_pulse).grid(row=3, column=0, sticky="w", pady=(8, 0))

        probe = ttk.LabelFrame(frame, text="Constant Voltage Current Probe", padding=10)
        probe.pack(fill="x", pady=(10, 0))
        self._add_entry(probe, "Voltage (V)", "PROBE_VOLTAGE", "0.1", 0, 0)
        self._add_entry(probe, "Duration (s)", "PROBE_DURATION", "1.0", 0, 2)
        self._add_entry(probe, "Sample Interval (s)", "PROBE_INTERVAL", "0.1", 1, 0)
        self._add_entry(probe, "Compliance (A)", "PROBE_COMP", "0.001", 1, 2)
        ttk.Checkbutton(probe, text="Use Current Autorange", variable=self._bool_var("PROBE_AUTORANGE", False)).grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Button(probe, text="Run Current Probe", command=self.queue_run_single_current_probe).grid(row=3, column=0, sticky="w", pady=(8, 0))

    def _create_protocol_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Protocol")
        left = ttk.Frame(frame)
        left.pack(side="left", fill="y")
        right = ttk.Frame(frame)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        ttk.Button(left, text="Add Step", command=self._add_protocol_step).pack(fill="x", pady=4)
        ttk.Button(left, text="Edit Selected", command=self._edit_protocol_step).pack(fill="x", pady=4)
        ttk.Button(left, text="Remove Selected", command=self._remove_protocol_step).pack(fill="x", pady=4)
        ttk.Button(left, text="Clear", command=self._clear_protocol).pack(fill="x", pady=4)
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        ttk.Button(left, text="Save Protocol", command=self._save_protocol).pack(fill="x", pady=4)
        ttk.Button(left, text="Load Protocol", command=self._load_protocol).pack(fill="x", pady=4)

        list_frame = ttk.Frame(right)
        list_frame.pack(fill="both", expand=True)
        self.protocol_listbox = tk.Listbox(list_frame, exportselection=False)
        self.protocol_listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.protocol_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.protocol_listbox.configure(yscrollcommand=scrollbar.set)

    def _create_log_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Output")
        self.output = scrolledtext.ScrolledText(frame, wrap="word")
        self.output.pack(fill="both", expand=True)

    def _create_live_plot_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Live Plot", padding=10)
        panel.pack(fill="both", expand=True)
        self.live_plot_figure = Figure(figsize=(4.2, 3.2), dpi=100)
        self.live_plot_axis = self.live_plot_figure.add_subplot(111)
        self._reset_live_plot("Live Sweep")
        self.live_plot_canvas = FigureCanvasTkAgg(self.live_plot_figure, master=panel)
        self.live_plot_canvas.draw_idle()
        self.live_plot_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _add_entry(self, parent, label, key, default, row, column):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", pady=(6, 0) if row else 0)
        ttk.Entry(parent, textvariable=self._string_var(key, default), width=14).grid(
            row=row, column=column + 1, sticky="ew", padx=(6, 10), pady=(6, 0) if row else 0
        )

    def _log(self, text):
        self.output.insert(tk.END, str(text))
        self.output.see(tk.END)

    def _choose_file(self, key):
        path = filedialog.askopenfilename(parent=self.root, filetypes=(("CSV Files", "*.csv"),))
        if path:
            self._string_var(key).set(path)

    def _choose_directory(self, key):
        path = filedialog.askdirectory(parent=self.root)
        if path:
            self._string_var(key).set(path)

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
        (self.live_plot_line,) = self.live_plot_axis.plot([], [], color="#0d6efd", linewidth=1.4)
        if hasattr(self, "live_plot_canvas"):
            self.live_plot_canvas.draw_idle()

    def _live_plot_callback(self, chunk, label=None):
        self.root.after(0, lambda: self._apply_live_plot_chunk(chunk, label))

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

    def _plot_saved_csv(self, csv_path, data_name):
        try:
            figure = self.data_handler.build_measurement_figure(csv_path, data_name=data_name, sample_id="crossbar", device_id=os.path.basename(csv_path))
        except Exception as exc:
            self._log(f"Could not rebuild saved plot: {exc}\n")
            return
        self.live_plot_figure.clf()
        src = figure.axes[0]
        axis = self.live_plot_figure.add_subplot(111)
        for line in src.get_lines():
            axis.plot(
                line.get_xdata(),
                line.get_ydata(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                label=line.get_label(),
            )
        axis.set_title(src.get_title())
        axis.set_xlabel(src.get_xlabel())
        axis.set_ylabel(src.get_ylabel())
        axis.set_xscale(src.get_xscale())
        axis.set_yscale(src.get_yscale())
        axis.grid(True, alpha=0.4)
        if src.get_legend() is not None:
            axis.legend()
        self.live_plot_axis = axis
        self.live_plot_canvas.draw_idle()

    def _save_current_graph(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".png", initialfile="crossbar_plot.png", filetypes=(("PNG Files", "*.png"), ("PDF Files", "*.pdf")))
        if path:
            self.live_plot_figure.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"Saved graph to {path}\n")

    def _augment_saved_metadata(self, saved_path, channels):
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
        payload.setdefault("gui_context", {}).update(
            {
                "operator": self._string_var("OPERATOR").get().strip(),
                "sample_id": self._string_var("SAMPLE").get().strip(),
                "notes": self.notes_text.get("1.0", tk.END).strip(),
                "channels": channels,
            }
        )
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def _draw_grid(self):
        self.grid_canvas.delete("all")
        cell = 36
        margin = 28
        for index in range(16):
            self.grid_canvas.create_text(margin + index * cell + cell / 2, 12, text=str(index + 1))
            self.grid_canvas.create_text(12, margin + index * cell + cell / 2, text=str(index + 1))
        for row in range(16):
            for col in range(16):
                x0 = margin + col * cell
                y0 = margin + row * cell
                rect = self.grid_canvas.create_rectangle(x0, y0, x0 + cell, y0 + cell, fill="white", outline="black")
                self.grid_rects[(row, col)] = rect
                self.selection_state[(row, col)] = False

    def _update_grid_color(self, row, col, selected):
        self.grid_canvas.itemconfig(self.grid_rects[(row, col)], fill="yellow" if selected else "white")
        current_state = self.selection_state.get((row, col), False)
        if selected and not current_state:
            self.selected_device_count += 1
        elif not selected and current_state:
            self.selected_device_count -= 1
        self.selection_state[(row, col)] = selected
        self._string_var("SELECTED_COUNT").set(f"Devices Selected: {self.selected_device_count}")

    def _on_grid_click(self, event):
        cell = 36
        margin = 28
        col = int((event.x - margin) // cell)
        row = int((event.y - margin) // cell)
        if 0 <= row < 16 and 0 <= col < 16:
            self._update_grid_color(row, col, not self.selection_state[(row, col)])

    def _select_all(self):
        for row in range(16):
            for col in range(16):
                self._update_grid_color(row, col, True)

    def _deselect_all(self):
        for row in range(16):
            for col in range(16):
                self._update_grid_color(row, col, False)

    def _select_half(self):
        for row in range(16):
            for col in range(16):
                self._update_grid_color(row, col, (row + col) % 2 == 0)

    def _toggle_selection(self):
        for row in range(16):
            for col in range(16):
                self._update_grid_color(row, col, not self.selection_state[(row, col)])

    def _get_selected_devices(self):
        devices = []
        for row in range(16):
            for col in range(16):
                if self.selection_state.get((row, col), False):
                    devices.append((row + 1, col + 1))
        return devices

    def _connect(self):
        self.crossbar.connect_multiplexer()
        self.crossbar.connect_SMU()
        if self.crossbar.is_mux_connected and self.crossbar.is_smu_connected:
            self._string_var("STATUS").set("Status: Connected (Keysight SMU)")
        else:
            self._string_var("STATUS").set("Status: Connection Failed")
        self._log("Connection sequence finished.\n")

    def _disconnect(self):
        self.crossbar.disconnect_multiplexer()
        self.crossbar.disconnect_SMU()
        self._string_var("STATUS").set("Status: Disconnected")
        self._log("Disconnected all crossbar hardware.\n")

    def _set_channels(self):
        try:
            ch1 = int(float(self._string_var("CH1").get()))
            ch2 = int(float(self._string_var("CH2").get()))
        except ValueError as exc:
            messagebox.showerror("Invalid Channels", str(exc), parent=self.root)
            return
        self.crossbar.switch_channels(ch1, ch2)

    def _run_in_background(self, label, callback):
        if self.worker and self.worker.is_alive():
            messagebox.showerror("Busy", "Another task is already running.", parent=self.root)
            return

        def runner():
            self._log(f"{label} started.\n")
            try:
                callback()
            except Exception as exc:
                self._log(f"{label} failed: {exc}\n")
            else:
                self._log(f"{label} finished.\n")

        self.worker = threading.Thread(target=runner, daemon=True)
        self.worker.start()

    def queue_run_single_dciv(self):
        self._run_in_background("Single DCIV", self._run_single_dciv)

    def _run_single_dciv(self):
        sweep_path = self._string_var("SWEEP_PATH").get()
        if not sweep_path or not os.path.exists(sweep_path):
            raise FileNotFoundError(f"Sweep file not found: {sweep_path}")
        ch1 = int(float(self._string_var("CH1").get()))
        ch2 = int(float(self._string_var("CH2").get()))
        self.root.after(0, lambda: self._reset_live_plot("Crossbar DCIV"))
        sweep_data, resistance_data, saved_path = self.crossbar.run_single_DCIV(
            ch1,
            ch2,
            pos_compl=float(self._string_var("DCIV_POS").get()),
            neg_compl=float(self._string_var("DCIV_NEG").get()),
            sweep_path=sweep_path,
            sweep_delay=float(self._string_var("DCIV_DELAY").get()) if self._string_var("DCIV_DELAY").get() else None,
            acq_delay=float(self._string_var("DCIV_ACQ").get()) if self._string_var("DCIV_ACQ").get() else None,
            plot=False,
            save=True,
            save_dir=self._string_var("SAVE_FOLDER").get(),
            use_4way_split=self._bool_var("DCIV_4WAY").get(),
            current_autorange=self._bool_var("DCIV_AUTORANGE").get(),
            include_read_probe=self._bool_var("DCIV_READ_PROBE").get(),
            progress_callback=self._live_plot_callback,
        )
        if saved_path:
            self._augment_saved_metadata(saved_path, {"ch1": ch1, "ch2": ch2})
            self._log(f"Saved sweep to {saved_path}\n")
        else:
            self._log("DCIV run did not save data.\n")

    def queue_run_single_pulse(self):
        self._run_in_background("Single Pulse", self._run_single_pulse)

    def _run_single_pulse(self):
        pulse_path = self._string_var("PULSE_PATH").get()
        if not pulse_path or not os.path.exists(pulse_path):
            raise FileNotFoundError(f"Pulse file not found: {pulse_path}")
        ch1 = int(float(self._string_var("CH1").get()))
        ch2 = int(float(self._string_var("CH2").get()))
        pulse_data, saved_path = self.crossbar.run_single_pulse(
            ch1,
            ch2,
            compliance=float(self._string_var("PULSE_COMP").get()),
            pulse_path=pulse_path,
            pulse_width=float(self._string_var("PULSE_WIDTH").get()) if self._string_var("PULSE_WIDTH").get() else None,
            plot=False,
            save=True,
            save_dir=self._string_var("SAVE_FOLDER").get(),
            set_acquire_delay=float(self._string_var("PULSE_ACQ").get()) if self._string_var("PULSE_ACQ").get() else None,
            current_autorange=self._bool_var("PULSE_AUTORANGE").get(),
        )
        if saved_path:
            self._augment_saved_metadata(saved_path, {"ch1": ch1, "ch2": ch2})
            self.root.after(0, lambda: self._plot_saved_csv(saved_path, "Pulse"))
            self._log(f"Saved pulse data to {saved_path}\n")
        else:
            self._log("Pulse run did not save data.\n")

    def queue_run_single_current_probe(self):
        self._run_in_background("Current Probe", self._run_single_current_probe)

    def _run_single_current_probe(self):
        ch1 = int(float(self._string_var("CH1").get()))
        ch2 = int(float(self._string_var("CH2").get()))
        self.root.after(0, lambda: self._reset_live_plot("Current Probe", x_label="Time (s)"))
        probe_data, saved_path = self.crossbar.run_single_current_probe(
            ch1,
            ch2,
            voltage=float(self._string_var("PROBE_VOLTAGE").get()),
            duration=float(self._string_var("PROBE_DURATION").get()),
            compliance=float(self._string_var("PROBE_COMP").get()),
            sample_interval=float(self._string_var("PROBE_INTERVAL").get()),
            plot=False,
            save=True,
            save_dir=self._string_var("SAVE_FOLDER").get(),
            current_autorange=self._bool_var("PROBE_AUTORANGE").get(),
            progress_callback=self._live_plot_callback,
        )
        if saved_path:
            self._augment_saved_metadata(saved_path, {"ch1": ch1, "ch2": ch2})
            self.root.after(0, lambda: self._plot_saved_csv(saved_path, "CurrentProbe"))
            self._log(f"Saved current probe data to {saved_path}\n")
        else:
            self._log("Current probe run did not save data.\n")

    def queue_measure_selected(self):
        selected = self._get_selected_devices()
        if not selected:
            messagebox.showerror("No Selection", "Select at least one grid cell first.", parent=self.root)
            return
        self._run_in_background("Grid Measurement", self._measure_selected)

    def _measure_selected(self):
        selected = self._get_selected_devices()
        save_dir = self._string_var("SAVE_FOLDER").get()
        action = self._string_var("GRID_ACTION", "DCIV").get()
        self._log(f"Starting {action} across {len(selected)} selected devices.\n")
        for ch1, ch2 in selected:
            self._log(f"Running on device {ch1}-{ch2}\n")
            if action == "DCIV":
                _, _, saved_path = self.crossbar.run_single_DCIV(
                    ch1,
                    ch2,
                    pos_compl=float(self._string_var("DCIV_POS").get()),
                    neg_compl=float(self._string_var("DCIV_NEG").get()),
                    sweep_path=self._string_var("SWEEP_PATH").get(),
                    sweep_delay=float(self._string_var("DCIV_DELAY").get()) if self._string_var("DCIV_DELAY").get() else None,
                    acq_delay=float(self._string_var("DCIV_ACQ").get()) if self._string_var("DCIV_ACQ").get() else None,
                    plot=False,
                    save=True,
                    save_dir=os.path.join(save_dir, f"device_{ch1}-{ch2}"),
                    use_4way_split=self._bool_var("DCIV_4WAY").get(),
                    current_autorange=self._bool_var("DCIV_AUTORANGE").get(),
                    include_read_probe=self._bool_var("DCIV_READ_PROBE").get(),
                    progress_callback=self._live_plot_callback,
                )
                if saved_path:
                    self._augment_saved_metadata(saved_path, {"ch1": ch1, "ch2": ch2})
            elif action == "PULSE":
                _, saved_path = self.crossbar.run_single_pulse(
                    ch1,
                    ch2,
                    compliance=float(self._string_var("PULSE_COMP").get()),
                    pulse_path=self._string_var("PULSE_PATH").get(),
                    pulse_width=float(self._string_var("PULSE_WIDTH").get()) if self._string_var("PULSE_WIDTH").get() else None,
                    plot=False,
                    save=True,
                    save_dir=os.path.join(save_dir, f"device_{ch1}-{ch2}"),
                    set_acquire_delay=float(self._string_var("PULSE_ACQ").get()) if self._string_var("PULSE_ACQ").get() else None,
                    current_autorange=self._bool_var("PULSE_AUTORANGE").get(),
                )
                if saved_path:
                    self._augment_saved_metadata(saved_path, {"ch1": ch1, "ch2": ch2})
            elif action == "CV_CURRENT_PROBE":
                _, saved_path = self.crossbar.run_single_current_probe(
                    ch1,
                    ch2,
                    voltage=float(self._string_var("PROBE_VOLTAGE").get()),
                    duration=float(self._string_var("PROBE_DURATION").get()),
                    compliance=float(self._string_var("PROBE_COMP").get()),
                    sample_interval=float(self._string_var("PROBE_INTERVAL").get()),
                    plot=False,
                    save=True,
                    save_dir=os.path.join(save_dir, f"device_{ch1}-{ch2}"),
                    current_autorange=self._bool_var("PROBE_AUTORANGE").get(),
                    progress_callback=self._live_plot_callback,
                )
                if saved_path:
                    self._augment_saved_metadata(saved_path, {"ch1": ch1, "ch2": ch2})
            else:
                self._run_protocol_for_device(ch1, ch2, os.path.join(save_dir, f"device_{ch1}-{ch2}"))
            time.sleep(0.1)

    def _run_protocol_for_device(self, ch1, ch2, save_dir):
        if not self.protocol_builder.protocol_list_configs:
            raise ValueError("Protocol is empty.")
        for step in self.protocol_builder.protocol_list_configs:
            step_type = step.get("type")
            params = dict(step.get("params", {}))
            if step_type == "DCIV":
                _, _, saved_path = self.crossbar.run_single_DCIV(
                    ch1,
                    ch2,
                    pos_compl=float(params.get("pos_compl", 0.001)),
                    neg_compl=float(params.get("neg_compl", 0.01)),
                    sweep_path=params.get("sweep_path"),
                    sweep_delay=params.get("sweep_delay"),
                    acq_delay=params.get("acq_delay"),
                    plot=False,
                    save=True,
                    save_dir=save_dir,
                    use_4way_split=bool(params.get("use_4way_split", True)),
                    current_autorange=bool(params.get("current_autorange", False)),
                    include_read_probe=bool(params.get("include_read_probe", True)),
                    progress_callback=self._live_plot_callback,
                )
                if saved_path:
                    self._augment_saved_metadata(saved_path, {"ch1": ch1, "ch2": ch2})
            elif step_type == "PULSE":
                _, saved_path = self.crossbar.run_single_pulse(
                    ch1,
                    ch2,
                    compliance=float(params.get("compliance", 0.01)),
                    pulse_path=params.get("pulse_path"),
                    pulse_width=params.get("pulse_width"),
                    plot=False,
                    save=True,
                    save_dir=save_dir,
                    set_acquire_delay=params.get("set_acquire_delay"),
                    current_autorange=bool(params.get("current_autorange", False)),
                )
                if saved_path:
                    self._augment_saved_metadata(saved_path, {"ch1": ch1, "ch2": ch2})
            elif step_type == "CV_CURRENT_PROBE":
                _, saved_path = self.crossbar.run_single_current_probe(
                    ch1,
                    ch2,
                    voltage=float(params.get("voltage", 0.1)),
                    duration=float(params.get("duration", 1.0)),
                    compliance=float(params.get("compliance", 0.001)),
                    sample_interval=float(params.get("sample_interval", 0.1)),
                    plot=False,
                    save=True,
                    save_dir=save_dir,
                    current_autorange=bool(params.get("current_autorange", False)),
                    progress_callback=self._live_plot_callback,
                )
                if saved_path:
                    self._augment_saved_metadata(saved_path, {"ch1": ch1, "ch2": ch2})
            elif step_type == "DELAY":
                time.sleep(float(params.get("duration", 1.0)))
            elif step_type == "LOG_MESSAGE":
                self._log(f"Protocol note for {ch1}-{ch2}: {params.get('message', '')}\n")
            else:
                raise ValueError(f"Unsupported crossbar protocol step '{step_type}'.")

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

    def on_close(self):
        try:
            self._disconnect()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    CrossbarTkApp().run()


if __name__ == "__main__":
    main()
