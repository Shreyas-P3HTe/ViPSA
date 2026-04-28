import inspect
import os
import sys
import types
import unittest
from unittest.mock import patch

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

try:
    import pyvisa  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pyvisa"] = types.SimpleNamespace(ResourceManager=None)


class _FakeResource:
    def __init__(self, address):
        self.address = address
        self.timeout = None
        self.read_termination = "\n"
        self.write_termination = "\n"

    def write(self, _command):
        return None

    def query(self, command):
        if command == "*IDN?":
            return "FAKE,MODEL,123,1.0"
        if command == ":ROUT:TERM?":
            return "FRON"
        if command.startswith(":MEAS:VOLT?"):
            return "0.0"
        if command.startswith(":MEAS:CURR?"):
            return "0.0"
        if command in {":READ?", "READ?"}:
            return "0.0,0.0"
        if command == ":SYST:ERR:ALL?":
            return "0,No error"
        return "0"

    def read(self):
        return "1"

    def close(self):
        return None


class _FakeResourceManager:
    def list_resources(self):
        return ("USB0::FAKE::INSTR",)

    def open_resource(self, address):
        return _FakeResource(address)


class ScpiDriverSkeletonTests(unittest.TestCase):
    def setUp(self):
        self.rm_patch = patch("pyvisa.ResourceManager", return_value=_FakeResourceManager())
        self.rm_patch.start()

    def tearDown(self):
        self.rm_patch.stop()

    def test_shared_scpi_contract_has_documented_placeholders(self):
        from vipsa.hardware.SCPI import SCPIInstrument, ScpiReading

        smu = SCPIInstrument(auto_connect=False)
        self.assertIsInstance(smu.read_current_at_voltage(0.1, 1e-6), ScpiReading)
        self.assertEqual(smu.source_voltage_measure_current([0.1], 1e-6), [])
        self.assertEqual(smu.source_current_measure_voltage([1e-6], 1.0), [])
        self.assertEqual(smu.run_voltage_pulse_train([0.1], 1e-6, 0.01), [])
        self.assertGreater(len(SCPIInstrument.source_voltage_measure_current.__doc__ or ""), 80)

    def test_keithley_skeleton_surface(self):
        from vipsa.hardware.keithley_2450 import Keithley2450, KeithleySMU

        self.assertTrue(issubclass(KeithleySMU, Keithley2450))
        smu = KeithleySMU(device_no=0, address="USB0::FAKE::INSTR")
        expected = {
            "initialize",
            "connect",
            "disconnect",
            "clear",
            "reset",
            "set_source_function",
            "set_sense_function",
            "set_source_range",
            "set_sense_range",
            "set_protection",
            "set_source_level",
            "set_output",
            "configure_measurement_format",
            "source_voltage_measure_current",
            "source_current_measure_voltage",
            "read_current_at_voltage",
            "measure_resistance",
            "run_voltage_pulse_train",
            "pulsed_measurement",
        }
        for name in expected:
            self.assertTrue(callable(getattr(smu, name, None)), name)
            self.assertGreater(len(getattr(type(smu), name).__doc__ or ""), 20, name)

    def test_keysight_skeleton_surface(self):
        from vipsa.hardware.keysight_b2902b import KeysightB2902B, KeysightSMU

        self.assertTrue(issubclass(KeysightSMU, KeysightB2902B))
        smu = KeysightSMU(device_no=0, address="USB0::FAKE::INSTR")
        expected = {
            "initialize",
            "connect",
            "disconnect",
            "clear",
            "reset",
            "set_source_function",
            "set_sense_function",
            "set_source_range",
            "set_sense_range",
            "set_protection",
            "set_source_level",
            "set_output",
            "configure_measurement_format",
            "source_voltage_measure_current",
            "source_current_measure_voltage",
            "read_current_at_voltage",
            "measure_resistance",
            "run_voltage_pulse_train",
            "pulsed_measurement",
            "response_dealer",
            "split_pulse_for_2_chan",
        }
        for name in expected:
            self.assertTrue(callable(getattr(smu, name, None)), name)
            self.assertGreater(len(getattr(type(smu), name).__doc__ or ""), 20, name)

    def test_constructor_signatures_remain_handler_friendly(self):
        from vipsa.hardware.keysight_b2902b import KeysightSMU
        from vipsa.hardware.keithley_2450 import KeithleySMU

        self.assertEqual(
            list(inspect.signature(KeysightSMU).parameters)[:5],
            ["device_no", "address", "switch", "switch_channel", "connect_switch"],
        )
        self.assertEqual(
            list(inspect.signature(KeithleySMU).parameters)[:5],
            ["device_no", "address", "switch", "switch_channel", "connect_switch"],
        )


if __name__ == "__main__":
    unittest.main()
