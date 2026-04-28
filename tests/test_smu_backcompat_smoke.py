import inspect
import importlib
import json
import os
import sys
import tempfile
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
            return "FRON"
        if command.startswith(":MEAS:VOLT?"):
            return "0.0"
        if command.startswith(":MEAS:CURR?"):
            return "0.0"
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


class _NullDriver:
    def __init__(self):
        self.address = "NULL::FAKE::INSTR"
        self.switch = None
        self.switch_channel = None
        self.switch_profile = None
        self.instrument_family = "null"
        self.supports_native_pulse = True

    def stop_output(self):
        return None

    def source_voltage_measure_current(self, voltages, current_compliance, delay_s=0.01, **_kwargs):
        records = []
        for index, voltage in enumerate(voltages):
            timestamp = round(index * float(delay_s), 12)
            records.append(
                {
                    "Time(T)": timestamp,
                    "Voltage (V)": float(voltage),
                    "Current (A)": 0.0,
                    "V_cmd (V)": float(voltage),
                    "I_cmd (A)": None,
                    "V_meas (V)": float(voltage),
                    "I_meas (A)": 0.0,
                    "V_error (V)": 0.0,
                    "I_error (A)": None,
                    "Cycle Number": 1.0,
                }
            )
        return records


class _DummyVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def _import_testmaker_module():
    os.environ["MPLBACKEND"] = "Agg"
    mpl_config_dir = os.path.join(PACKAGE_ROOT, "tests_artifacts_tmp", "mplconfig")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_config_dir
    return importlib.import_module("vipsa.gui.Testmaker_tk")


class _HeadlessTestmakerHarness:
    _DEFAULT_KEY_MAP = {
        "forward_voltage": "SW_FORWARD_V",
        "reset_voltage": "SW_RESET_V",
        "step_voltage": "SW_STEP_V",
        "timer_delay": "SW_DELAY",
        "cycles": "SW_CYCLES",
        "forming_cycle": "SW_FORMING",
        "forming_voltage": "SW_FORMING_V",
        "peak_hold_steps": "SW_HOLD",
        "return_to_zero": "SW_RETURN_ZERO",
        "sweep_mode": "SW_MODE",
        "pos_compl": "SW_POS_COMPL",
        "neg_compl": "SW_NEG_COMPL",
        "use_4way_split": "SW_4WAY",
        "base_width": "PU_BASE",
        "write_voltage": "PU_WRITE_V",
        "write_width": "PU_WRITE_W",
        "write_gap": "PU_WRITE_G",
        "write_pulses": "PU_WRITE_N",
        "read_voltage": "PU_READ_V",
        "read_width": "PU_READ_W",
        "read_gap": "PU_READ_G",
        "read_pulses": "PU_READ_N",
        "erase_voltage": "PU_ERASE_V",
        "erase_width": "PU_ERASE_W",
        "erase_gap": "PU_ERASE_G",
        "erase_pulses": "PU_ERASE_N",
        "pulse_cycles": "PU_CYCLES",
        "cycle_gap": "PU_CYCLE_GAP",
        "initial_gap": "PU_INITIAL_GAP",
        "final_read_block": "PU_FINAL_READ",
        "pulse_compliance": "PU_COMPL",
        "set_acquire_delay": "PU_ACQ_DELAY",
    }
    _BOOL_KEYS = {
        "SW_FORMING",
        "SW_RETURN_ZERO",
        "SW_4WAY",
        "PU_FINAL_READ",
        "-PROTO_ALIGN-",
        "-PROTO_APPROACH-",
    }

    def __init__(self, module, test_name):
        self._module = module
        self._selected_test = module.TEST_BY_NAME[test_name]
        self.generated = None
        self.last_saved_csv_path = ""
        self.protocol_callback = None
        self.close_callback = None
        self.root = None
        self.vars = {}
        self.widgets = {}
        self._bool_var("-PROTO_ALIGN-", False).set(False)
        self._bool_var("-PROTO_APPROACH-", False).set(False)
        self._load_defaults(dict(self._selected_test.get("defaults", {})))

    def _string_var(self, key, default):
        var = self.vars.get(key)
        if var is None:
            var = _DummyVar(str(default))
            self.vars[key] = var
        return var

    def _bool_var(self, key, default):
        var = self.vars.get(key)
        if var is None:
            var = _DummyVar(bool(default))
            self.vars[key] = var
        return var

    def _get_selected_test(self):
        return self._selected_test

    def _load_defaults(self, defaults):
        for source_key, value in defaults.items():
            target = self._DEFAULT_KEY_MAP.get(source_key)
            if target is None:
                continue
            if target in self._BOOL_KEYS:
                self._bool_var(target, False).set(bool(value))
            else:
                self._string_var(target, "").set(str(value))

    def _get_sweep_params(self):
        return self._module.TestmakerApp._get_sweep_params(self)

    def _get_pulse_params(self):
        return self._module.TestmakerApp._get_pulse_params(self)

    def generate(self):
        return self._module.TestmakerApp._generate_for_selected_test(self)

    def build_protocol_steps_for_path(self, csv_path, generated):
        return self._module.TestmakerApp._build_protocol_steps_for_path(self, csv_path, generated)

    def _build_protocol_steps_for_path(self, csv_path, generated):
        return self.build_protocol_steps_for_path(csv_path, generated)

    def _build_protocol_steps(self):
        return self._module.TestmakerApp._build_protocol_steps(self)

    def export_protocol_json(self):
        return self._module.TestmakerApp._export_protocol_json(self)


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

    def test_import_all_refactor_modules_and_wrap_null_driver(self):
        modules = [
            "vipsa.hardware.keithley_2450",
            "vipsa.hardware.keysight_b2902b",
            "vipsa.hardware.keithley_707b",
            "vipsa.hardware.Source_Measure_Unit",
        ]
        for module_name in modules:
            self.assertIsNotNone(importlib.import_module(module_name), module_name)
        self.assertIsNotNone(_import_testmaker_module())

        from vipsa.hardware.Source_Measure_Unit import SourceMeasureUnit

        smu = SourceMeasureUnit(_NullDriver())
        records = smu.source_voltage_measure_current([0.1, 0.2], current_compliance=1e-3, delay_s=0.01)

        self.assertEqual(len(records), 2)
        self.assertEqual(smu.records_to_legacy_array(records).shape, (2, 3))
        self.assertIsNone(smu.stop_output())

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

        self.assertEqual(switch.calls, [("connect_named_route", "keysight")])

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

        self.assertEqual(switch.calls, [("connect_named_route", "keithley")])

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
        self.assertIn(":ROUT:TERM FRON", writes)
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

    def test_testmaker_catalog_entries_have_required_skill_fields(self):
        testmaker = _import_testmaker_module()
        required_fields = {
            "name",
            "category",
            "status",
            "mode",
            "generator",
            "protocol_type",
            "description",
            "feasibility",
            "defaults",
            "composition",
            "validation",
            "outputs",
        }

        for entry in testmaker.TEST_CATALOG:
            self.assertTrue(required_fields.issubset(entry), entry["name"])
            self.assertIn(entry["protocol_type"], {"DCIV", "PULSE", "INFO"}, entry["name"])

            composition = entry["composition"]
            self.assertIn("steps", composition, entry["name"])
            self.assertIn("executable", composition, entry["name"])

            if entry["mode"] == testmaker.MODE_INFO:
                self.assertEqual(entry["protocol_type"], "INFO", entry["name"])
                self.assertEqual(composition["protocol_family"], "INFO", entry["name"])
                self.assertFalse(composition["executable"], entry["name"])

    def test_testmaker_dciv_generation_and_protocol_export_smoke(self):
        testmaker = _import_testmaker_module()
        app = _HeadlessTestmakerHarness(testmaker, "Current-Voltage (I-V) Sweep")
        app._bool_var("-PROTO_ALIGN-", False).set(True)
        app._bool_var("-PROTO_APPROACH-", False).set(True)

        generated = app.generate()
        self.assertEqual(generated["protocol_type"], "DCIV")
        self.assertGreater(len(generated["times"]), 0)
        self.assertEqual(len(generated["times"]), len(generated["voltages"]))

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "dciv.csv")
            json_path = os.path.join(tmpdir, "dciv_protocol.json")
            testmaker.save_csv(list(generated["times"]), list(generated["voltages"]), csv_path)
            app.generated = generated
            app.last_saved_csv_path = csv_path

            with patch.object(testmaker.filedialog, "asksaveasfilename", return_value=json_path):
                with patch.object(testmaker.messagebox, "showinfo", return_value=None):
                    app.export_protocol_json()

            with open(json_path, "r", encoding="utf-8") as handle:
                steps = json.load(handle)

        self.assertEqual([step["type"] for step in steps], ["ALIGN", "APPROACH", "DCIV"])
        self.assertEqual(steps[-1]["params"]["sweep_path"], csv_path)
        self.assertEqual(
            set(steps[-1]["params"]),
            {"sweep_path", "pos_compl", "neg_compl", "sweep_delay", "align", "approach", "smu_select", "use_4way_split"},
        )

    def test_testmaker_pulse_generation_and_protocol_export_smoke(self):
        testmaker = _import_testmaker_module()
        app = _HeadlessTestmakerHarness(testmaker, "Endurance Cycling")

        generated = app.generate()
        self.assertEqual(generated["protocol_type"], "PULSE")
        self.assertGreater(len(generated["times"]), 0)
        self.assertEqual(len(generated["times"]), len(generated["voltages"]))

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "pulse.csv")
            json_path = os.path.join(tmpdir, "pulse_protocol.json")
            testmaker.save_csv(list(generated["times"]), list(generated["voltages"]), csv_path)
            app.generated = generated
            app.last_saved_csv_path = csv_path

            with patch.object(testmaker.filedialog, "asksaveasfilename", return_value=json_path):
                with patch.object(testmaker.messagebox, "showinfo", return_value=None):
                    app.export_protocol_json()

            with open(json_path, "r", encoding="utf-8") as handle:
                steps = json.load(handle)

        self.assertEqual([step["type"] for step in steps], ["PULSE"])
        self.assertEqual(steps[-1]["params"]["pulse_path"], csv_path)
        self.assertEqual(
            set(steps[-1]["params"]),
            {"pulse_path", "compliance", "pulse_width", "align", "approach", "smu_select", "set_acquire_delay"},
        )


if __name__ == "__main__":
    unittest.main()
