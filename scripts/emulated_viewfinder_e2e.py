#!/usr/bin/env python3
"""Run a GUI-level Viewfinder protocol against simulated equipment.

This intentionally exercises the active Tk GUI and workflow stack. The only
mocked boundary is equipment construction: the GUI still opens, imports a
protocol, queues connection, and runs a grid protocol through the normal worker
thread machinery.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vipsa-mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import pandas as pd
import tkinter as tk


REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK_PATH = REPO_ROOT / "archive" / "simulation_2026-04-15" / "hardware" / "mock_hardware.py"
ARTIFACT_ROOT = REPO_ROOT / "archive" / "simulation_2026-04-15" / "runs" / f"live_gui_e2e_{time.strftime('%Y%m%d_%H%M%S')}"
INPUT_DIR = ARTIFACT_ROOT / "inputs"
OUTPUT_DIR = ARTIFACT_ROOT / "output"
LOG_PATH = ARTIFACT_ROOT / "run.log"
PROTOCOL_PATH = INPUT_DIR / "advanced_protocol.json"
SWEEP_PATH = INPUT_DIR / "dciv_sweep.csv"
PULSE_PATH = INPUT_DIR / "pulse_train.csv"
GRID_PATH = INPUT_DIR / "grid.csv"


def load_mock_hardware():
	spec = importlib.util.spec_from_file_location("vipsa_e2e_mock_hardware", MOCK_PATH)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Could not load mock hardware from {MOCK_PATH}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def write_inputs() -> list[dict]:
	INPUT_DIR.mkdir(parents=True, exist_ok=True)
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	sweep_rows = []
	timestamp = 0.0
	for voltage in [0.0, 0.05, 0.1, 0.05, 0.0, -0.05, -0.1, -0.05, 0.0] * 2:
		sweep_rows.append({"Time": round(timestamp, 6), "Voltage": voltage})
		timestamp += 0.001
	pd.DataFrame(sweep_rows).to_csv(SWEEP_PATH, index=False)

	pulse_rows = []
	timestamp = 0.0
	for voltage in [1.0, 0.1, 0.0, -1.0, 0.1, 0.0] * 4:
		pulse_rows.append({"Time": round(timestamp, 6), "Voltage": voltage})
		timestamp += 0.001
	pd.DataFrame(pulse_rows).to_csv(PULSE_PATH, index=False)

	pd.DataFrame(
		[
			{"Device": 1, "X": 0.0, "Y": 0.0},
			{"Device": 2, "X": 20.0, "Y": 0.0},
		]
	).to_csv(GRID_PATH, index=False)

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
				"delay": 0.001,
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
				"read_probe_mode": "between_segments",
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
	]
	with open(PROTOCOL_PATH, "w", encoding="utf-8") as handle:
		json.dump(protocol, handle, indent=2)
	return protocol


def current_rss_bytes() -> int:
	try:
		with open("/proc/self/status", "r", encoding="utf-8") as handle:
			for line in handle:
				if line.startswith("VmRSS:"):
					return int(line.split()[1]) * 1024
	except Exception:
		pass
	return 0


def collect_artifacts() -> dict:
	csv_paths = sorted(OUTPUT_DIR.rglob("*.csv"))
	metadata_paths = sorted(OUTPUT_DIR.rglob("*.metadata.json"))
	png_paths = sorted(OUTPUT_DIR.rglob("*.png"))
	metadata_types = []
	for path in metadata_paths:
		with open(path, "r", encoding="utf-8") as handle:
			payload = json.load(handle)
		metadata_types.append(
			{
				"path": str(path),
				"data_name": payload.get("data_name"),
				"step": payload.get("protocol_context", {}).get("current_step_type"),
			}
		)
	return {
		"csv_count": len(csv_paths),
		"metadata_count": len(metadata_paths),
		"png_count": len(png_paths),
		"csv_paths": [str(path) for path in csv_paths],
		"metadata_types": metadata_types,
	}


def main() -> int:
	if str(REPO_ROOT) not in sys.path:
		sys.path.insert(0, str(REPO_ROOT))

	protocol = write_inputs()
	mock_hardware = load_mock_hardware()

	from vipsa.gui import Viewfinder4_tk as viewfinder
	from vipsa.gui.Viewfinder4_tk import RedirectText, VipsaGUI

	def build_mock_smu(self, label):
		key, display = self._normalize_smu_label(label)
		address = self.discovered_instruments.get(key)
		smu = mock_hardware.build_mock_smu(
			display,
			switch=self.switch,
			stage=getattr(self.vipsa, "stage", None),
			address=address,
		)
		self._update_smu_stage_reference(smu)
		return smu

	def connect_mock_equipment(self, selected_smu):
		if self.is_equipment_connected:
			self._disconnect_all(log_message=False)

		setup = mock_hardware.build_mock_setup(
			arduino_port=self._get_value("-ARDUINO_PORT-", "SIM::COM5"),
			arduino_baud=int(float(self._get_value("-ARDUINO_BAUD-", "115200"))),
			arduino_scale=float(self._get_value("-ARDUINO_SCALE-", "1")),
			zaber_port=self._get_value("-ZABER_PORT-", "SIM::COM7"),
		)
		self.discovered_instruments = dict(setup["discovered_instruments"])
		self.vipsa.top_light = setup["top_light"]
		self.vipsa.stage = setup["stage"]
		self.vipsa.Zaber = setup["Zaber"]
		self.vipsa.zaber_x = setup["zaber_x"]
		self.vipsa.zaber_y = setup["zaber_y"]
		self.vipsa.SMU_name = selected_smu
		self.vipsa.equipment = True
		self.switch = setup["switch"]
		self.smus = {}
		self.active_smu_label = None
		self.is_equipment_connected = True
		self._set_manual_active_smu(selected_smu, force_reconnect=True)
		self._set_status(f"Connected mock equipment ({selected_smu})", viewfinder.SUCCESS_COLOR)
		self._log("Connected simulated switch, Keithley, Keysight, Arduino stage, Zaber stages, and light.\n")
		self._refresh_setup_monitor()

	viewfinder.VipsaGUI._build_smu = build_mock_smu
	viewfinder.VipsaGUI._connect_real_equipment = connect_mock_equipment

	ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
	start = time.monotonic()
	max_rss = {"value": 0}
	stop_metrics = threading.Event()
	status = {"value": "FAIL", "reason": ""}
	live_chunks = {"count": 0}

	with open(LOG_PATH, "w", encoding="utf-8") as log_handle:
		real_stdout = sys.__stdout__

		def log_line(text: str) -> None:
			line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text.rstrip()}\n"
			log_handle.write(line)
			log_handle.flush()
			real_stdout.write(line)
			real_stdout.flush()

		def metrics_loop() -> None:
			while not stop_metrics.is_set():
				rss = current_rss_bytes()
				max_rss["value"] = max(max_rss["value"], rss)
				log_handle.write(
					f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] METRIC elapsed_s={time.monotonic() - start:.3f} rss_mb={rss / (1024 * 1024):.3f}\n"
				)
				log_handle.flush()
				stop_metrics.wait(1.0)

		metrics_thread = threading.Thread(target=metrics_loop, daemon=True)
		metrics_thread.start()

		root = tk.Tk()
		app = VipsaGUI(root)
		root.title("ViPSA Live GUI Emulated E2E")

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

		original_live = app._live_plot_callback

		def live_callback(chunk, label=None):
			live_chunks["count"] += 1
			original_live(chunk, label=label)

		app._live_plot_callback = live_callback

		def wait_for_idle(phase, on_done, timeout_s):
			deadline = time.monotonic() + timeout_s

			def poll():
				alive = bool(app.worker_thread and app.worker_thread.is_alive())
				if not alive:
					log_line(f"ACTION {phase}_done")
					on_done()
					return
				if time.monotonic() > deadline:
					status["reason"] = f"timeout waiting for {phase}"
					finish()
					return
				root.after(200, poll)

			root.after(200, poll)

		def start_protocol():
			log_line(f"ACTION import_protocol {PROTOCOL_PATH}")
			if not app.protocol_builder.import_protocol(str(PROTOCOL_PATH)):
				status["reason"] = "protocol import failed"
				finish()
				return
			app._refresh_protocol_display()
			display_steps = app.protocol_builder.get_protocol_display_list()
			log_line("ASSERT protocol_steps=" + json.dumps(display_steps))
			app._set_value("-MEAS_SAVE_FOLDER-", str(OUTPUT_DIR))
			app._set_value("-GRID_MEAS_PATH-", str(GRID_PATH))
			app._set_value("-GRID_SAMPLE_ID-", "e2e_sample")
			app._set_value("-GRID_STARTPOINT-", "1")
			app._set_value("-GRID_SKIP-", "1")
			app._set_value("-GRID_RANDOMIZE-", False)
			app._set_value("-ACTIVE_SMU-", "Keithley2450")
			log_line("ACTION queue_grid_protocol")
			app._queue_grid_measurement("PROTOCOL")
			wait_for_idle("grid_protocol", finish, timeout_s=120)

		def after_connect():
			log_line("ASSERT gui_connected=true")
			start_protocol()

		def start_connect():
			app._set_value("-ACTIVE_SMU-", "Keithley2450")
			app._set_value("-ARDUINO_PORT-", "SIM::COM5")
			app._set_value("-ZABER_PORT-", "SIM::COM7")
			app._set_value("-MEAS_SAVE_FOLDER-", str(OUTPUT_DIR))
			log_line("ACTION queue_connect_all")
			app._queue_connect_all()
			wait_for_idle("connect", after_connect, timeout_s=30)

		def finish():
			if stop_metrics.is_set():
				return
			artifacts = collect_artifacts()
			step_statuses = []
			for item in artifacts["metadata_types"]:
				if item["step"]:
					step_statuses.append(item["step"])
			expected_data = {"Sweep", "Pulse", "Resistance"}
			observed_data = {item["data_name"] for item in artifacts["metadata_types"]}
			if not status["reason"]:
				if artifacts["csv_count"] < 6:
					status["reason"] = f"expected at least 6 CSVs, saw {artifacts['csv_count']}"
				elif not expected_data.issubset(observed_data):
					status["reason"] = f"missing expected data types: {sorted(expected_data - observed_data)}"
				elif "DCIV" not in step_statuses or "PULSE" not in step_statuses:
					status["reason"] = f"missing protocol metadata steps in {step_statuses}"
				else:
					status["value"] = "PASS"

			log_line(f"RESULT status={status['value']}")
			if status["reason"]:
				log_line(f"RESULT reason={status['reason']}")
			log_line(f"RESULT total_elapsed_s={time.monotonic() - start:.3f}")
			log_line(f"RESULT max_rss_mb={max_rss['value'] / (1024 * 1024):.3f}")
			log_line(f"RESULT live_plot_chunks={live_chunks['count']}")
			log_line("RESULT protocol=" + json.dumps(protocol))
			log_line("RESULT artifacts=" + json.dumps(artifacts, indent=2))
			stop_metrics.set()
			try:
				app.on_close()
			except Exception:
				root.destroy()

		root.after(500, start_connect)
		root.mainloop()
		stop_metrics.set()
		metrics_thread.join(timeout=2.0)

	return 0 if status["value"] == "PASS" else 1


if __name__ == "__main__":
	raise SystemExit(main())
