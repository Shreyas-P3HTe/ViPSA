import json
import os
import shutil
import sys
import unittest

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from vipsa.analysis.Datahandling import Data_Handler
from vipsa.analysis.sweep_generation import generate_voltage_data


class SweepGenerationTests(unittest.TestCase):
    def test_positive_first_matches_legacy_default_shape(self):
        times, voltages, cycles = generate_voltage_data(
            forward_voltage=1.0,
            reset_voltage=-1.0,
            step_voltage=1.0,
            timer_delay=0.1,
            forming_cycle=False,
            forming_voltage=None,
            cycles=1,
            sweep_mode="positive_first",
        )
        self.assertEqual(voltages, [0.0, 1.0, 0.0, 0.0, -1.0, 0.0])
        self.assertEqual(len(times), len(voltages))
        self.assertEqual(cycles, [1, 1, 1, 1, 1, 1])

    def test_negative_first_starts_negative(self):
        _times, voltages, cycles = generate_voltage_data(
            forward_voltage=1.0,
            reset_voltage=-1.0,
            step_voltage=1.0,
            timer_delay=0.1,
            forming_cycle=False,
            forming_voltage=None,
            cycles=2,
            sweep_mode="negative_first",
        )
        self.assertEqual(voltages[:6], [0.0, -1.0, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(cycles[0], 1)
        self.assertEqual(cycles[-1], 2)

    def test_positive_to_negative_crosses_zero_without_duplicate_endpoint(self):
        _times, voltages, _cycles = generate_voltage_data(
            forward_voltage=1.0,
            reset_voltage=-1.0,
            step_voltage=0.5,
            timer_delay=0.1,
            forming_cycle=False,
            forming_voltage=None,
            cycles=1,
            return_to_zero=False,
            sweep_mode="positive_to_negative",
        )
        self.assertEqual(voltages, [1.0, 0.5, 0.0, -0.5, -1.0])

    def test_negative_only_range_is_supported(self):
        _times, voltages, _cycles = generate_voltage_data(
            forward_voltage=0.0,
            reset_voltage=-1.0,
            step_voltage=0.5,
            timer_delay=0.1,
            forming_cycle=False,
            forming_voltage=None,
            cycles=1,
            return_to_zero=False,
            sweep_mode="negative_first",
        )
        self.assertIn(-1.0, voltages)
        self.assertEqual(voltages[0], 0.0)


class ArtifactSaveTests(unittest.TestCase):
    def test_save_file_writes_csv_png_and_metadata_sidecar(self):
        handler = Data_Handler()
        tmpdir = os.path.join(PACKAGE_ROOT, "tests_artifacts_tmp")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            records = [
                {
                    "Time(T)": 0.0,
                    "Voltage (V)": 0.0,
                    "Current (A)": 1e-9,
                    "V_cmd (V)": 0.0,
                    "V_meas (V)": 0.0,
                },
                {
                    "Time(T)": 0.1,
                    "Voltage (V)": 0.5,
                    "Current (A)": 1e-6,
                    "V_cmd (V)": 0.5,
                    "V_meas (V)": 0.49,
                },
            ]
            csv_path = handler.save_file(
                records,
                "Sweep",
                "sample_a",
                "device_1",
                cont_current=1e-9,
                Z_pos=12.3,
                save_directory=tmpdir,
                metadata={"test_case": "artifact_save"},
            )

            base_path, _ = os.path.splitext(csv_path)
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(f"{base_path}.png"))
            self.assertTrue(os.path.exists(f"{base_path}.metadata.json"))

            with open(f"{base_path}.metadata.json", "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["test_case"], "artifact_save")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_sigmoid_pulse_output_rebuilds_probability_plot(self):
        handler = Data_Handler()
        tmpdir = os.path.join(PACKAGE_ROOT, "tests_artifacts_tmp")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            manifest_path = os.path.join(tmpdir, "sigmoid_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "mode": "randomized_set_voltage_sigmoid",
                        "read_voltage": 0.1,
                        "reset_voltage": -0.5,
                        "set_voltage_points": [0.2, 0.6],
                        "set_voltage_sequence": [0.2, 0.6],
                        "loops": 2,
                        "write_pulses": 1,
                        "read_pulses_per_block": 1,
                        "erase_pulses": 1,
                    },
                    handle,
                    indent=2,
                )

            records = [
                {"Time(T)": 0.0, "Voltage (V)": 0.0, "Current (A)": 1e-9, "V_cmd (V)": 0.0},
                {"Time(T)": 0.1, "Voltage (V)": 0.2, "Current (A)": 4e-7, "V_cmd (V)": 0.2},
                {"Time(T)": 0.2, "Voltage (V)": 0.1, "Current (A)": 1e-7, "V_cmd (V)": 0.1},
                {"Time(T)": 0.3, "Voltage (V)": -0.5, "Current (A)": 3e-7, "V_cmd (V)": -0.5},
                {"Time(T)": 0.4, "Voltage (V)": 0.1, "Current (A)": 2e-7, "V_cmd (V)": 0.1},
                {"Time(T)": 0.5, "Voltage (V)": 0.0, "Current (A)": 1e-9, "V_cmd (V)": 0.0},
                {"Time(T)": 0.6, "Voltage (V)": 0.6, "Current (A)": 8e-7, "V_cmd (V)": 0.6},
                {"Time(T)": 0.7, "Voltage (V)": 0.1, "Current (A)": 1e-5, "V_cmd (V)": 0.1},
                {"Time(T)": 0.8, "Voltage (V)": -0.5, "Current (A)": 4e-7, "V_cmd (V)": -0.5},
                {"Time(T)": 0.9, "Voltage (V)": 0.1, "Current (A)": 1e-7, "V_cmd (V)": 0.1},
            ]

            csv_path = handler.save_file(
                records,
                "Pulse",
                "sample_sigmoid",
                "device_7",
                cont_current=1e-9,
                Z_pos=4.2,
                save_directory=tmpdir,
                metadata={
                    "measurement_parameters": {
                        "pulse_mode": "randomized_set_voltage_sigmoid",
                        "sigmoid_manifest_path": manifest_path,
                    },
                    "protocol_context": {
                        "current_step_params": {
                            "pulse_mode": "randomized_set_voltage_sigmoid",
                            "sigmoid_manifest_path": manifest_path,
                        }
                    },
                },
            )

            figure = handler.build_measurement_figure(
                csv_path,
                data_name="Pulse",
                sample_id="sample_sigmoid",
                device_id="device_7",
                managed=False,
            )
            axis = figure.axes[0]
            line = axis.get_lines()[0]

            self.assertEqual(axis.get_ylabel(), "Switching probability")
            self.assertEqual(list(line.get_xdata()), [0.2, 0.6])
            self.assertEqual(list(line.get_ydata()), [0.0, 1.0])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
