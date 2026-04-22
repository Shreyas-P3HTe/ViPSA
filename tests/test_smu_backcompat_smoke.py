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
        self.writes = []

    def write(self, command):
        self.writes.append(command)

    def query(self, command):
        self.writes.append(command)
        if command == "*IDN?":
            return "FAKE,MODEL,123,1.0"
        if command == ":ROUT:TERM?":
            return "REAR"
        if command == ":READ?" or command == "READ?":
            return "0.0,0.0"
        if command == ":SYST:ERR:ALL?":
            return "0,No error"
        if command == "*OPC?":
            return "1"
        if command.startswith(":FETC:ARR:SOUR"):
            return "0.1,0.2,0.3"
        if command.startswith(":FETC:ARR:VOLT"):
            return "0.101,0.202,0.303"
        if command.startswith(":FETC:ARR:CURR"):
            return "1e-06,2e-06,3e-06"
        if command.startswith(":FETC:ARR:TIME"):
            return "0,0.05,0.1"
        if command.startswith(":TRAC:ACT?"):
            return "3"
        if command.startswith(":TRAC:DATA?"):
            return "0.1,1e-06,0,0.2,2e-06,0.05,0.3,3e-06,0.1"
        return "0"

    def read(self):
        return "1"

    def close(self):
        return None


class _FakeResourceManager:
    def list_resources(self):
        return ("USB0::FAKE::INSTR", "TCPIP0::FAKE::INSTR")

    def open_resource(self, address):
        return _FakeResource(address)


class _FakeSwitch:
    def __init__(self):
        self.calls = []

    def connect_named_route(self, name):
        self.calls.append(("connect_named_route", name))

    def open_all(self):
        self.calls.append(("open_all",))

    def close_channel(self, route):
        self.calls.append(("close_channel", route))

    def open_channel(self, route):
        self.calls.append(("open_channel", route))


class SmuBackCompatSmokeTests(unittest.TestCase):
    def setUp(self):
        self.rm_patch = patch("pyvisa.ResourceManager", return_value=_FakeResourceManager())
        self.rm_patch.start()

    def tearDown(self):
        self.rm_patch.stop()

    def test_facade_imports_still_work(self):
        from vipsa.hardware.Source_Measure_Unit import KeysightSMU, KeithleySMU, Keithley707B, pyvisa

        self.assertEqual(KeysightSMU.__name__, "KeysightSMU")
        self.assertEqual(KeithleySMU.__name__, "KeithleySMU")
        self.assertEqual(Keithley707B.__name__, "Keithley707B")
        self.assertTrue(hasattr(pyvisa, "ResourceManager"))

    def test_old_constructor_signatures_still_match(self):
        from vipsa.hardware.Source_Measure_Unit import KeysightSMU, KeithleySMU

        keysight_params = list(inspect.signature(KeysightSMU).parameters)
        keithley_params = list(inspect.signature(KeithleySMU).parameters)
        self.assertEqual(
            keysight_params[:5],
            ["device_no", "address", "switch", "switch_channel", "connect_switch"],
        )
        self.assertEqual(
            keithley_params[:5],
            ["device_no", "address", "switch", "switch_channel", "connect_switch"],
        )

    def test_keysight_old_api_shape_and_switch_hooks(self):
        from vipsa.hardware.Source_Measure_Unit import KeysightSMU

        switch = _FakeSwitch()
        smu = KeysightSMU(
            device_no=0,
            address="USB0::FAKE::INSTR",
            switch=switch,
            switch_channel="keysight",
            connect_switch=True,
        )

        expected = {
            "connect_switch_path",
            "disconnect_switch_path",
            "close_session",
            "get_address",
            "get_error",
            "reset_device",
            "sync",
            "send_command",
            "write",
            "ask",
            "prepare_contact_probe",
            "stop_output",
            "abort_measurement",
            "get_contact_current",
            "get_contact_current_fast",
            "split_list",
            "split_list_by_4",
            "simple_IV_sweep",
            "list_IV_sweep_manual",
            "scan_read_vlist",
            "list_IV_sweep_split",
            "list_IV_sweep_split_4",
            "pulsed_measurement",
            "response_dealer",
            "split_pulse_for_2_chan",
            "general_channel_pulsing",
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
            "source_voltage_measure_current",
            "source_current_measure_voltage",
            "read_current_at_voltage",
            "measure_resistance",
            "run_voltage_pulse_train",
        }
        for name in expected:
            self.assertTrue(callable(getattr(smu, name, None)), name)

        self.assertEqual(switch.calls, [])

    def test_keithley_old_api_shape_and_switch_hooks(self):
        from vipsa.hardware.Source_Measure_Unit import KeithleySMU

        switch = _FakeSwitch()
        smu = KeithleySMU(
            device_no=0,
            address="USB0::FAKE::INSTR",
            switch=switch,
            switch_channel="keithley",
            connect_switch=True,
        )

        expected = {
            "connect_switch_path",
            "disconnect_switch_path",
            "close_session",
            "get_address",
            "reset_device",
            "write",
            "ask",
            "prepare_contact_probe",
            "stop_output",
            "abort_measurement",
            "get_contact_current",
            "get_contact_current_fast",
            "run_read_probe",
            "identify_linear_segments",
            "split_by_polarity",
            "split_sweep_by_4",
            "list_IV_sweep_split_4",
            "run_linear_segment",
            "list_IV_sweep_split",
            "pulsed_measurement",
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
            "source_voltage_measure_current",
            "source_current_measure_voltage",
            "read_current_at_voltage",
            "measure_resistance",
            "run_voltage_pulse_train",
        }
        for name in expected:
            self.assertTrue(callable(getattr(smu, name, None)), name)

        self.assertEqual(switch.calls, [])

    def test_keithley_voltage_sweep_uses_native_source_list(self):
        from vipsa.hardware.Source_Measure_Unit import KeithleySMU

        smu = KeithleySMU(device_no=0, address="USB0::FAKE::INSTR")
        smu.smu.writes.clear()

        records = smu.source_voltage_measure_current(
            voltages=[0.1, 0.2, 0.3],
            current_compliance=1e-5,
            delay_s=0.05,
            current_range=1e-5,
            use_auto_current_range=False,
        )

        writes = smu.smu.writes
        self.assertEqual(len(records), 3)
        self.assertIn(":ROUT:TERM REAR", writes)
        self.assertIn(":SOUR:FUNC VOLT", writes)
        self.assertIn(":SOUR:VOLT:READ:BACK ON", writes)
        self.assertIn(":SENS:CURR:RANG:AUTO OFF", writes)
        self.assertIn(":SENS:CURR:RANG 1e-05", writes)
        self.assertIn(":SOUR:LIST:VOLT 0.1,0.2,0.3", writes)
        self.assertIn(':SOUR:SWE:VOLT:LIST 1, 0.05, 1, OFF, "defbuffer1"', writes)
        self.assertIn("*OPC?", writes)
        self.assertIn(':TRAC:DATA? 1, 3, "defbuffer1", SOUR, READ, REL', writes)
        self.assertFalse(any(command.startswith(":SOUR:VOLT:LEV ") for command in writes))
        self.assertEqual(records[0]["V_meas (V)"], 0.1)

    def test_keysight_voltage_sweep_uses_native_list_and_reads_voltage_current(self):
        from vipsa.hardware.Source_Measure_Unit import KeysightSMU

        smu = KeysightSMU(device_no=0, address="USB0::FAKE::INSTR")
        smu.smu.writes.clear()

        records = smu.source_voltage_measure_current(
            voltages=[0.1, 0.2, 0.3],
            current_compliance=1e-5,
            delay_s=0.05,
            current_range=1e-5,
            use_auto_current_range=False,
        )

        writes = smu.smu.writes
        self.assertEqual(len(records), 3)
        self.assertIn(":ROUT:TERM REAR", writes)
        self.assertIn(":FORM:ELEM:SENS VOLT,CURR,TIME,SOUR", writes)
        self.assertIn(":SOUR1:FUNC:MODE VOLT", writes)
        self.assertIn(":SOUR1:VOLT:MODE LIST", writes)
        self.assertIn(':SENS1:FUNC "CURR","VOLT"', writes)
        self.assertIn(":SENS1:VOLT:RANG:AUTO ON", writes)
        self.assertIn(":SENS1:CURR:RANG:AUTO OFF", writes)
        self.assertIn(":SENS1:CURR:RANG 1e-05", writes)
        self.assertIn(":SOUR1:LIST:VOLT 0.1,0.2,0.3", writes)
        self.assertIn(":SOUR1:SWE:POIN 3", writes)
        self.assertIn(":SOUR1:SWE:RANG BEST", writes)
        self.assertIn(":TRIG1:SOUR TIM", writes)
        self.assertIn(":TRIG1:TIM 0.05", writes)
        self.assertIn(":TRIG1:COUN 3", writes)
        self.assertIn(":TRAC1:CLE", writes)
        self.assertIn(":FETC:ARR:VOLT? (@1)", writes)
        self.assertIn(":FETC:ARR:CURR? (@1)", writes)
        self.assertEqual(records[0]["V_cmd (V)"], 0.1)
        self.assertEqual(records[0]["V_meas (V)"], 0.101)
        self.assertEqual(records[0]["Current (A)"], 1e-6)


if __name__ == "__main__":
    unittest.main()
