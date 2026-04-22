# -*- coding: utf-8 -*-
"""
Enhanced standalone list maker for voltage sweeps and pulse trains.
"""

import csv
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import PySimpleGUI as sg
except ModuleNotFoundError:
    sg = None


EPSILON = 1e-9
PLOT_STYLE = {
    "color": "#1768AC",
    "markerfacecolor": "#F26419",
    "markeredgecolor": "#F26419",
    "linewidth": 1.8,
    "markersize": 3.5,
}
SWEEP_KEYS = (
    "forward_voltage",
    "reset_voltage",
    "step_voltage",
    "timer_delay",
    "cycles",
    "forming_cycle",
    "forming_voltage",
    "peak_hold_steps",
    "return_to_zero",
)
PULSE_KEYS = (
    "write_pulses",
    "write_voltage",
    "write_width",
    "write_gap",
    "read_pulses",
    "read_voltage",
    "read_width",
    "read_gap",
    "erase_pulses",
    "erase_voltage",
    "erase_width",
    "erase_gap",
    "pulse_cycles",
    "final_read_block",
    "cycle_gap",
)


def parse_float(value, field_name, allow_blank=False, default=None):
    text = str(value).strip()
    if text == "":
        if allow_blank:
            return default
        raise ValueError(f"{field_name} is required.")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number.") from exc


def parse_int(value, field_name, minimum=None, allow_blank=False, default=None):
    text = str(value).strip()
    if text == "":
        if allow_blank:
            return default
        raise ValueError(f"{field_name} is required.")
    try:
        number = int(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if minimum is not None and number < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return number


def inclusive_ramp(start, end, step):
    if step <= 0:
        raise ValueError("Step voltage must be greater than zero.")
    direction = 1 if end >= start else -1
    actual_step = step * direction
    values = []
    current = float(start)

    while True:
        values.append(round(current, 12))
        current += actual_step
        if direction > 0 and current > end + EPSILON:
            break
        if direction < 0 and current < end - EPSILON:
            break

    if abs(values[-1] - end) > EPSILON:
        values.append(round(float(end), 12))
    return values


def add_hold_points(sequence, hold_value, hold_steps):
    if hold_steps <= 0:
        return sequence
    return sequence + [round(float(hold_value), 12)] * hold_steps


def build_bidirectional_cycle(forward_voltage, reset_voltage, step_voltage, peak_hold_steps=0):
    positive_up = inclusive_ramp(0.0, forward_voltage, step_voltage)
    positive_down = positive_up[-2::-1] if len(positive_up) > 1 else []
    positive_leg = add_hold_points(positive_up, forward_voltage, peak_hold_steps) + positive_down

    negative_down = inclusive_ramp(0.0, reset_voltage, step_voltage)
    negative_up = negative_down[-2::-1] if len(negative_down) > 1 else []
    negative_leg = add_hold_points(negative_down, reset_voltage, peak_hold_steps) + negative_up

    return positive_leg + negative_leg


def generate_voltage_data(
    forward_voltage,
    reset_voltage,
    step_voltage,
    timer_delay,
    forming_cycle,
    forming_voltage,
    cycles,
    peak_hold_steps=0,
    return_to_zero=True,
):
    voltages = []
    times = []
    current_time = 0.0

    if forming_cycle:
        forming_sequence = build_bidirectional_cycle(
            forming_voltage,
            reset_voltage,
            step_voltage,
            peak_hold_steps=peak_hold_steps,
        )
        for voltage in forming_sequence:
            voltages.append(voltage)
            times.append(round(current_time, 12))
            current_time += timer_delay

    for _ in range(cycles):
        cycle_sequence = build_bidirectional_cycle(
            forward_voltage,
            reset_voltage,
            step_voltage,
            peak_hold_steps=peak_hold_steps,
        )
        for voltage in cycle_sequence:
            voltages.append(voltage)
            times.append(round(current_time, 12))
            current_time += timer_delay

    if return_to_zero and voltages and abs(voltages[-1]) > EPSILON:
        voltages.append(0.0)
        times.append(round(current_time, 12))

    return times, voltages


def generate_pulsing_data(
    write_pulses,
    write_voltage,
    write_width,
    read_pulses,
    read_voltage,
    read_width,
    erase_pulses,
    erase_voltage,
    erase_width,
    cycles,
    write_gap=None,
    read_gap=None,
    erase_gap=None,
    final_read_block=True,
    cycle_gap=0.0,
):
    voltages = []
    times = []
    current_time = 0.0

    write_gap = write_width if write_gap is None else write_gap
    read_gap = read_width if read_gap is None else read_gap
    erase_gap = erase_width if erase_gap is None else erase_gap

    def append_pulse_block(count, voltage, width, gap):
        nonlocal current_time
        for _ in range(count):
            voltages.append(voltage)
            times.append(round(current_time, 12))
            current_time += width
            voltages.append(0.0)
            times.append(round(current_time, 12))
            current_time += gap

    for cycle_index in range(cycles):
        append_pulse_block(write_pulses, write_voltage, write_width, write_gap)
        append_pulse_block(read_pulses, read_voltage, read_width, read_gap)
        append_pulse_block(erase_pulses, erase_voltage, erase_width, erase_gap)
        if final_read_block:
            append_pulse_block(read_pulses, read_voltage, read_width, read_gap)

        if cycle_gap > 0 and cycle_index < cycles - 1:
            voltages.append(0.0)
            times.append(round(current_time, 12))
            current_time += cycle_gap

    return times, voltages


def save_to_csv(times, voltages, filename):
    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Time (s)", "Voltage (V)"])
        for time_value, voltage_value in zip(times, voltages):
            csvwriter.writerow([time_value, voltage_value])


def build_summary(times, voltages, label):
    if not times or not voltages:
        return f"{label}\nNo data generated yet."

    duration = times[-1] if len(times) > 1 else 0.0
    positive_points = sum(1 for voltage in voltages if voltage > EPSILON)
    negative_points = sum(1 for voltage in voltages if voltage < -EPSILON)
    zero_points = len(voltages) - positive_points - negative_points

    return "\n".join(
        (
            label,
            f"Points: {len(voltages)}",
            f"Duration: {duration:.6f} s",
            f"Voltage window: {min(voltages):.4f} V to {max(voltages):.4f} V",
            f"Positive / Negative / Zero points: {positive_points} / {negative_points} / {zero_points}",
        )
    )


def plot_data(times, voltages, title):
    figure, axis = plt.subplots(figsize=(8.2, 4.6), dpi=100)
    axis.step(times, voltages, where="post", marker="o", **PLOT_STYLE)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Voltage (V)")
    axis.set_title(title)
    axis.grid(True, alpha=0.35)
    figure.tight_layout()
    return figure


def draw_figure(canvas, figure):
    for child in canvas.winfo_children():
        child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


class EnhancedListmakerApp:
    def __init__(self):
        if sg is None:
            raise ImportError("PySimpleGUI is required to launch the enhanced listmaker GUI.")
        sg.theme("SystemDefault")
        self.generated_data = {"sweep": None, "pulse": None}
        self.figure_refs = {"sweep": None, "pulse": None}
        self.window = sg.Window(
            "Enhanced Voltage Sweep Generator",
            self.build_layout(),
            finalize=True,
            resizable=True,
            size=(1080, 760),
        )
        self.toggle_forming_voltage(False)

    def build_layout(self):
        iv_sweep_layout = [
            [
                sg.Frame(
                    "Sweep Parameters",
                    [
                        [sg.Text("Forward voltage (V)", size=(20, 1)), sg.Input("1.0", key="forward_voltage", size=(12, 1))],
                        [sg.Text("Reset voltage (V)", size=(20, 1)), sg.Input("-1.0", key="reset_voltage", size=(12, 1))],
                        [sg.Text("Step voltage (V)", size=(20, 1)), sg.Input("0.05", key="step_voltage", size=(12, 1))],
                        [sg.Text("Timer delay (s)", size=(20, 1)), sg.Input("0.001", key="timer_delay", size=(12, 1))],
                        [sg.Text("Cycles", size=(20, 1)), sg.Input("1", key="cycles", size=(12, 1))],
                        [
                            sg.Checkbox(
                                "Enable forming cycle",
                                key="forming_cycle",
                                default=False,
                                enable_events=True,
                            )
                        ],
                        [
                            sg.Text("Forming voltage (V)", size=(20, 1)),
                            sg.Input("2.0", key="forming_voltage", size=(12, 1), disabled=True),
                        ],
                    ],
                    expand_x=True,
                ),
                sg.Frame(
                    "Advanced Sweep Controls",
                    [
                        [sg.Text("Peak hold steps", size=(20, 1)), sg.Input("0", key="peak_hold_steps", size=(12, 1))],
                        [sg.Checkbox("Force final 0 V point", key="return_to_zero", default=True)],
                        [sg.Button("Preview Sweep", key="Visualize"), sg.Button("Save Sweep CSV", key="Save Sweep")],
                        [sg.Button("Load Sweep Preset"), sg.Button("Save Sweep Preset")],
                    ],
                    expand_x=True,
                ),
            ],
            [
                sg.Multiline(
                    "",
                    key="sweep_summary",
                    size=(50, 6),
                    disabled=True,
                    autoscroll=False,
                    expand_x=True,
                )
            ],
            [sg.Canvas(key="canvas", expand_x=True, expand_y=True, background_color="white")],
        ]

        pulsing_layout = [
            [
                sg.Frame(
                    "Write / Read / Erase Settings",
                    [
                        [
                            sg.Text("Write pulses", size=(12, 1)),
                            sg.Input("1", key="write_pulses", size=(8, 1)),
                            sg.Text("Voltage (V)", size=(10, 1)),
                            sg.Input("1.5", key="write_voltage", size=(8, 1)),
                            sg.Text("Width (s)", size=(9, 1)),
                            sg.Input("0.001", key="write_width", size=(8, 1)),
                            sg.Text("Gap (s)", size=(8, 1)),
                            sg.Input("", key="write_gap", size=(8, 1)),
                        ],
                        [
                            sg.Text("Read pulses", size=(12, 1)),
                            sg.Input("1", key="read_pulses", size=(8, 1)),
                            sg.Text("Voltage (V)", size=(10, 1)),
                            sg.Input("0.1", key="read_voltage", size=(8, 1)),
                            sg.Text("Width (s)", size=(9, 1)),
                            sg.Input("0.001", key="read_width", size=(8, 1)),
                            sg.Text("Gap (s)", size=(8, 1)),
                            sg.Input("", key="read_gap", size=(8, 1)),
                        ],
                        [
                            sg.Text("Erase pulses", size=(12, 1)),
                            sg.Input("1", key="erase_pulses", size=(8, 1)),
                            sg.Text("Voltage (V)", size=(10, 1)),
                            sg.Input("-2.0", key="erase_voltage", size=(8, 1)),
                            sg.Text("Width (s)", size=(9, 1)),
                            sg.Input("0.001", key="erase_width", size=(8, 1)),
                            sg.Text("Gap (s)", size=(8, 1)),
                            sg.Input("", key="erase_gap", size=(8, 1)),
                        ],
                    ],
                    expand_x=True,
                )
            ],
            [
                sg.Frame(
                    "Pulse Sequence Options",
                    [
                        [sg.Text("Cycles", size=(14, 1)), sg.Input("10", key="pulse_cycles", size=(10, 1))],
                        [sg.Text("Cycle gap (s)", size=(14, 1)), sg.Input("0", key="cycle_gap", size=(10, 1))],
                        [sg.Checkbox("Add read block after erase", key="final_read_block", default=True)],
                        [sg.Button("Preview Pulses", key="Generate Pulses"), sg.Button("Save Pulse CSV", key="Save Pulses")],
                        [sg.Button("Load Pulse Preset"), sg.Button("Save Pulse Preset")],
                    ],
                    expand_x=True,
                )
            ],
            [
                sg.Multiline(
                    "",
                    key="pulse_summary",
                    size=(50, 6),
                    disabled=True,
                    autoscroll=False,
                    expand_x=True,
                )
            ],
            [sg.Canvas(key="pulsing_canvas", expand_x=True, expand_y=True, background_color="white")],
        ]

        return [
            [
                sg.TabGroup(
                    [[sg.Tab("I-V Sweep", iv_sweep_layout), sg.Tab("Pulsing", pulsing_layout)]],
                    expand_x=True,
                    expand_y=True,
                )
            ]
        ]

    def toggle_forming_voltage(self, enabled):
        self.window["forming_voltage"].update(disabled=not enabled)

    def update_summary(self, mode, text):
        summary_key = "sweep_summary" if mode == "sweep" else "pulse_summary"
        self.window[summary_key].update(value=text)

    def render_plot(self, mode, times, voltages):
        canvas_key = "canvas" if mode == "sweep" else "pulsing_canvas"
        title = "I-V Sweep Preview" if mode == "sweep" else "Pulse Train Preview"
        previous_figure = self.figure_refs[mode]
        if previous_figure is not None:
            plt.close(previous_figure)
        figure = plot_data(times, voltages, title)
        self.figure_refs[mode] = figure
        draw_figure(self.window[canvas_key].TKCanvas, figure)

    def validate_sweep_inputs(self, values):
        forward_voltage = parse_float(values["forward_voltage"], "Forward voltage")
        reset_voltage = parse_float(values["reset_voltage"], "Reset voltage")
        step_voltage = parse_float(values["step_voltage"], "Step voltage")
        timer_delay = parse_float(values["timer_delay"], "Timer delay")
        cycles = parse_int(values["cycles"], "Cycles", minimum=1)
        peak_hold_steps = parse_int(values["peak_hold_steps"], "Peak hold steps", minimum=0)
        forming_cycle = bool(values["forming_cycle"])
        forming_voltage = None

        if forward_voltage <= 0:
            raise ValueError("Forward voltage must be greater than zero.")
        if reset_voltage >= 0:
            raise ValueError("Reset voltage must be negative.")
        if step_voltage <= 0:
            raise ValueError("Step voltage must be greater than zero.")
        if timer_delay <= 0:
            raise ValueError("Timer delay must be greater than zero.")

        if forming_cycle:
            forming_voltage = parse_float(values["forming_voltage"], "Forming voltage")
            if forming_voltage <= 0:
                raise ValueError("Forming voltage must be greater than zero.")
            if forming_voltage < forward_voltage:
                raise ValueError("Forming voltage should be at least as large as the forward voltage.")

        return {
            "forward_voltage": forward_voltage,
            "reset_voltage": reset_voltage,
            "step_voltage": step_voltage,
            "timer_delay": timer_delay,
            "forming_cycle": forming_cycle,
            "forming_voltage": forming_voltage,
            "cycles": cycles,
            "peak_hold_steps": peak_hold_steps,
            "return_to_zero": bool(values["return_to_zero"]),
        }

    def validate_pulse_inputs(self, values):
        parsed = {
            "write_pulses": parse_int(values["write_pulses"], "Write pulses", minimum=0),
            "write_voltage": parse_float(values["write_voltage"], "Write voltage"),
            "write_width": parse_float(values["write_width"], "Write width"),
            "write_gap": parse_float(values["write_gap"], "Write gap", allow_blank=True, default=None),
            "read_pulses": parse_int(values["read_pulses"], "Read pulses", minimum=0),
            "read_voltage": parse_float(values["read_voltage"], "Read voltage"),
            "read_width": parse_float(values["read_width"], "Read width"),
            "read_gap": parse_float(values["read_gap"], "Read gap", allow_blank=True, default=None),
            "erase_pulses": parse_int(values["erase_pulses"], "Erase pulses", minimum=0),
            "erase_voltage": parse_float(values["erase_voltage"], "Erase voltage"),
            "erase_width": parse_float(values["erase_width"], "Erase width"),
            "erase_gap": parse_float(values["erase_gap"], "Erase gap", allow_blank=True, default=None),
            "cycles": parse_int(values["pulse_cycles"], "Pulse cycles", minimum=1),
            "final_read_block": bool(values["final_read_block"]),
            "cycle_gap": parse_float(values["cycle_gap"], "Cycle gap"),
        }

        if all(
            parsed[count_key] == 0
            for count_key in ("write_pulses", "read_pulses", "erase_pulses")
        ):
            raise ValueError("At least one pulse block must contain one or more pulses.")

        for field_name in ("write_width", "read_width", "erase_width", "cycle_gap"):
            if parsed[field_name] < 0:
                raise ValueError(f"{field_name.replace('_', ' ').title()} cannot be negative.")

        for field_name in ("write_gap", "read_gap", "erase_gap"):
            if parsed[field_name] is not None and parsed[field_name] < 0:
                raise ValueError(f"{field_name.replace('_', ' ').title()} cannot be negative.")

        return parsed

    def preview_sweep(self, values):
        params = self.validate_sweep_inputs(values)
        times, voltages = generate_voltage_data(**params)
        self.generated_data["sweep"] = {"times": times, "voltages": voltages}
        self.update_summary("sweep", build_summary(times, voltages, "Sweep preview"))
        self.render_plot("sweep", times, voltages)

    def preview_pulses(self, values):
        params = self.validate_pulse_inputs(values)
        times, voltages = generate_pulsing_data(
            params["write_pulses"],
            params["write_voltage"],
            params["write_width"],
            params["read_pulses"],
            params["read_voltage"],
            params["read_width"],
            params["erase_pulses"],
            params["erase_voltage"],
            params["erase_width"],
            params["cycles"],
            write_gap=params["write_gap"],
            read_gap=params["read_gap"],
            erase_gap=params["erase_gap"],
            final_read_block=params["final_read_block"],
            cycle_gap=params["cycle_gap"],
        )
        self.generated_data["pulse"] = {"times": times, "voltages": voltages}
        self.update_summary("pulse", build_summary(times, voltages, "Pulse preview"))
        self.render_plot("pulse", times, voltages)

    def save_generated_csv(self, mode):
        data = self.generated_data[mode]
        if not data:
            label = "sweep" if mode == "sweep" else "pulse"
            sg.popup_error(f"No {label} data to save. Generate a preview first.")
            return

        save_path = sg.popup_get_file(
            "Save as",
            save_as=True,
            no_window=True,
            default_extension=".csv",
            file_types=(("CSV Files", "*.csv"),),
        )
        if not save_path:
            return
        save_to_csv(data["times"], data["voltages"], save_path)
        sg.popup_no_wait(f"Saved CSV to:\n{save_path}")

    def save_preset(self, mode):
        keys = SWEEP_KEYS if mode == "sweep" else PULSE_KEYS
        preset_data = {
            "mode": mode,
            "values": {key: self.window[key].get() for key in keys},
        }

        save_path = sg.popup_get_file(
            "Save preset",
            save_as=True,
            no_window=True,
            default_extension=".json",
            file_types=(("JSON Files", "*.json"),),
        )
        if not save_path:
            return

        with open(save_path, "w", encoding="utf-8") as preset_file:
            json.dump(preset_data, preset_file, indent=2)
        sg.popup_no_wait(f"Preset saved to:\n{save_path}")

    def load_preset(self, mode):
        preset_path = sg.popup_get_file(
            "Load preset",
            no_window=True,
            file_types=(("JSON Files", "*.json"),),
        )
        if not preset_path:
            return

        with open(preset_path, "r", encoding="utf-8") as preset_file:
            preset_data = json.load(preset_file)

        preset_mode = preset_data.get("mode")
        if preset_mode != mode:
            raise ValueError(
                f"This preset is for {preset_mode or 'an unknown mode'}, not {mode}."
            )

        keys = SWEEP_KEYS if mode == "sweep" else PULSE_KEYS
        values = preset_data.get("values", {})
        for key in keys:
            if key in values:
                self.window[key].update(values[key])

        if mode == "sweep":
            self.toggle_forming_voltage(bool(self.window["forming_cycle"].get()))

        sg.popup_no_wait(f"Loaded {mode} preset from:\n{preset_path}")

    def run(self):
        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break

            try:
                if event == "forming_cycle":
                    self.toggle_forming_voltage(bool(values["forming_cycle"]))
                elif event == "Visualize":
                    self.preview_sweep(values)
                elif event == "Generate Pulses":
                    self.preview_pulses(values)
                elif event == "Save Sweep":
                    self.save_generated_csv("sweep")
                elif event == "Save Pulses":
                    self.save_generated_csv("pulse")
                elif event == "Save Sweep Preset":
                    self.save_preset("sweep")
                elif event == "Load Sweep Preset":
                    self.load_preset("sweep")
                elif event == "Save Pulse Preset":
                    self.save_preset("pulse")
                elif event == "Load Pulse Preset":
                    self.load_preset("pulse")
            except Exception as exc:
                sg.popup_error(str(exc))

        for figure in self.figure_refs.values():
            if figure is not None:
                plt.close(figure)
        self.window.close()


if __name__ == "__main__":
    EnhancedListmakerApp().run()
