#!/usr/bin/env python3

import json
import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import tkinter as tk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vipsa.gui.Viewfinder4_tk import RedirectText, VipsaGUI


LOG_PATH = REPO_ROOT / "simulation_run.log"
ARTIFACT_DIR = REPO_ROOT / "stress_sim_artifacts"
SAVE_DIR = ARTIFACT_DIR / "output"
PROTOCOL_PATH = ARTIFACT_DIR / "complex_protocol.json"
SWEEP_PATH = ARTIFACT_DIR / "complex_sweep.csv"
PULSE_PATH = ARTIFACT_DIR / "complex_pulse.csv"
GRID_PATH = ARTIFACT_DIR / "grid.csv"


def current_rss_bytes():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except Exception:
        pass
    return 0


def build_artifacts():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    sweep_rows = []
    t = 0.0
    sweep_voltages = [
        0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0,
        -0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0,
    ]
    for cycle in range(5):
        for voltage in sweep_voltages:
            sweep_rows.append({"Time": round(t, 6), "Voltage": voltage})
            t += 0.002
    pd.DataFrame(sweep_rows).to_csv(SWEEP_PATH, index=False)

    pulse_pattern = [1.2, 0.1, 0.0, -1.1, 0.1, 0.0, 1.0, 0.1, 0.0, -0.9, 0.1, 0.0]
    pulse_rows = []
    t = 0.0
    for _ in range(18):
        for voltage in pulse_pattern:
            pulse_rows.append({"Time": round(t, 6), "Voltage": voltage})
            t += 0.001
    pd.DataFrame(pulse_rows).to_csv(PULSE_PATH, index=False)

    grid_rows = [
        {"Device": 1, "X": 0.0, "Y": 0.0},
        {"Device": 2, "X": 20.0, "Y": 0.0},
        {"Device": 3, "X": 40.0, "Y": 0.0},
        {"Device": 4, "X": 0.0, "Y": 20.0},
        {"Device": 5, "X": 20.0, "Y": 20.0},
        {"Device": 6, "X": 40.0, "Y": 20.0},
    ]
    pd.DataFrame(grid_rows).to_csv(GRID_PATH, index=False)

    protocol = [
        {
            "type": "APPROACH",
            "params": {
                "smu_select": "Keithley2450",
                "step_size": 0.5,
                "test_voltage": 0.05,
                "lower_threshold": 1e-11,
                "upper_threshold": 5e-11,
                "max_attempts": 16,
                "delay": 0.01,
            },
        },
        {
            "type": "DCIV",
            "params": {
                "sweep_path": str(SWEEP_PATH),
                "pos_compl": 0.001,
                "neg_compl": 0.001,
                "sweep_delay": 0.0005,
                "align": False,
                "approach": False,
                "smu_select": "Keithley2450",
                "use_4way_split": True,
                "include_read_probe": True,
                "current_autorange": False,
            },
        },
        {
            "type": "PULSE",
            "params": {
                "pulse_path": str(PULSE_PATH),
                "compliance": 0.01,
                "pulse_width": 0.001,
                "align": False,
                "approach": False,
                "smu_select": "KeysightB2901BL",
                "set_acquire_delay": 0.0005,
                "current_autorange": False,
            },
        },
        {
            "type": "APPROACH",
            "params": {
                "smu_select": "KeysightB2901BL",
                "step_size": 0.5,
                "test_voltage": 0.1,
                "lower_threshold": 1e-11,
                "upper_threshold": 5e-11,
                "max_attempts": 16,
                "delay": 0.01,
            },
        },
        {
            "type": "PULSE",
            "params": {
                "pulse_path": str(PULSE_PATH),
                "compliance": 0.01,
                "pulse_width": 0.001,
                "align": False,
                "approach": False,
                "smu_select": "KeysightB2901BL",
                "set_acquire_delay": 0.0005,
                "current_autorange": False,
            },
        },
        {
            "type": "DCIV",
            "params": {
                "sweep_path": str(SWEEP_PATH),
                "pos_compl": 0.001,
                "neg_compl": 0.001,
                "sweep_delay": 0.0005,
                "align": False,
                "approach": False,
                "smu_select": "Keithley2450",
                "use_4way_split": False,
                "include_read_probe": True,
                "current_autorange": False,
                "read_probe_mode": "between_segments",
            },
        },
    ]
    with open(PROTOCOL_PATH, "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)


def main():
    build_artifacts()
    os.environ["VIPSA_SIMULATE_HARDWARE"] = "1"

    with open(LOG_PATH, "w", encoding="utf-8") as log_handle:
        start_time = time.monotonic()
        stop_metrics = threading.Event()
        metrics = {"max_rss": 0}

        def log_line(text):
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{stamp}] {text}"
            if not line.endswith("\n"):
                line += "\n"
            log_handle.write(line)
            log_handle.flush()

        def metrics_loop():
            while not stop_metrics.is_set():
                rss = current_rss_bytes()
                metrics["max_rss"] = max(metrics["max_rss"], rss)
                elapsed = time.monotonic() - start_time
                log_line(f"METRIC elapsed_s={elapsed:.3f} rss_mb={rss / (1024 * 1024):.3f}")
                stop_metrics.wait(1.0)

        metrics_thread = threading.Thread(target=metrics_loop, daemon=True)
        metrics_thread.start()

        root = tk.Tk()
        app = VipsaGUI(root)
        root.title("ViPSA Stress Simulation")

        original_log = app._log

        def tee_log(text):
            clean = str(text).rstrip("\n")
            if clean:
                for line in clean.splitlines():
                    log_line(f"APP {line}")
            original_log(text)

        app._log = tee_log
        app.protocol_builder.logger = tee_log
        sys.stdout = RedirectText(tee_log)
        sys.stderr = RedirectText(tee_log)

        original_live_callback = app._live_plot_callback

        def logged_live_callback(chunk, label=None):
            try:
                rows = list(chunk.tolist()) if hasattr(chunk, "tolist") else list(chunk)
            except Exception:
                rows = []
            if rows:
                log_line(
                    "DATA "
                    + json.dumps(
                        {
                            "label": label,
                            "rows": rows,
                        }
                    )
                )
            original_live_callback(chunk, label=label)

        app._live_plot_callback = logged_live_callback

        phase_times = {}

        def wait_for_idle(phase_name, on_done, timeout_s):
            deadline = time.monotonic() + timeout_s

            def poll():
                worker_alive = bool(app.worker_thread and app.worker_thread.is_alive())
                if not worker_alive:
                    phase_times[f"{phase_name}_end"] = time.monotonic()
                    on_done()
                    return
                if time.monotonic() > deadline:
                    log_line(f"FAIL phase_timeout={phase_name}")
                    finish("FAIL")
                    return
                root.after(200, poll)

            root.after(200, poll)

        def start_protocol_run():
            log_line(f"ACTION load_protocol path={PROTOCOL_PATH}")
            if not app.protocol_builder.import_protocol(str(PROTOCOL_PATH)):
                log_line("FAIL protocol_import")
                finish("FAIL")
                return
            app._refresh_protocol_display()
            app._set_value("-MEAS_SAVE_FOLDER-", str(SAVE_DIR))
            app._set_value("-GRID_MEAS_PATH-", str(GRID_PATH))
            app._set_value("-GRID_SAMPLE_ID-", "stress_sample")
            app._set_value("-GRID_STARTPOINT-", "1")
            app._set_value("-GRID_SKIP-", "1")
            app._set_value("-GRID_RANDOMIZE-", False)
            app._set_value("-ACTIVE_SMU-", "Keithley2450")
            phase_times["grid_start"] = time.monotonic()
            log_line("ACTION run_grid_protocol")
            app._queue_grid_measurement("PROTOCOL")
            wait_for_idle("grid", finish_ok, timeout_s=180)

        def after_connect():
            log_line("ACTION connect_done")
            start_protocol_run()

        def start_connect():
            app._set_value("-SIMULATE_HARDWARE-", True)
            app._set_value("-ACTIVE_SMU-", "Keithley2450")
            app._set_value("-ARDUINO_PORT-", "SIM::COM5")
            app._set_value("-ZABER_PORT-", "SIM::COM7")
            app._set_value("-MEAS_SAVE_FOLDER-", str(SAVE_DIR))
            phase_times["connect_start"] = time.monotonic()
            log_line("ACTION connect_mock_equipment")
            app._queue_connect_all()
            wait_for_idle("connect", after_connect, timeout_s=60)

        def finish_ok():
            finish("PASS")

        def finish(status):
            if stop_metrics.is_set():
                return
            total_elapsed = time.monotonic() - start_time
            csv_outputs = sorted(str(path) for path in SAVE_DIR.rglob("*.csv"))
            log_line(f"RESULT status={status}")
            log_line(f"RESULT total_elapsed_s={total_elapsed:.3f}")
            if "connect_start" in phase_times and "connect_end" in phase_times:
                log_line(
                    f"RESULT connect_elapsed_s={phase_times['connect_end'] - phase_times['connect_start']:.3f}"
                )
            if "grid_start" in phase_times and "grid_end" in phase_times:
                log_line(f"RESULT grid_elapsed_s={phase_times['grid_end'] - phase_times['grid_start']:.3f}")
            log_line(f"RESULT max_rss_mb={metrics['max_rss'] / (1024 * 1024):.3f}")
            log_line("RESULT output_csv=" + json.dumps(csv_outputs))
            stop_metrics.set()
            try:
                app.on_close()
            except Exception:
                root.destroy()

        root.after(500, start_connect)
        root.mainloop()
        stop_metrics.set()
        metrics_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
