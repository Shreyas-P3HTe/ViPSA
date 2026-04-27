import os
import sys
import unittest
from unittest.mock import patch

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS_ROOT = os.path.abspath(os.path.dirname(__file__))
for path in (PACKAGE_ROOT, TESTS_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from fake_visa import FakeVisaRM


def _contains(log, text):
    return any(text in command for command in log)


class CommandSequenceTests(unittest.TestCase):
    def test_keithley_basic_init_sends_reset_commands(self):
        from vipsa.hardware.keithley_2450 import KeithleySMU

        fake_rm = FakeVisaRM()
        with patch("pyvisa.ResourceManager", return_value=fake_rm):
            smu = KeithleySMU(address="USB0::FAKE::INSTR")
            try:
                log = smu.smu.get_command_log()
                self.assertIn("*RST", log)
                self.assertIn("*CLS", log)
                self.assertIn(":SOUR:FUNC VOLT", log)
                self.assertIn(':SENS:FUNC "CURR"', log)
                self.assertIn(":OUTP OFF", log)
            finally:
                smu.close_session()

    def test_keithley_get_contact_current_sequence(self):
        from vipsa.hardware.keithley_2450 import KeithleySMU

        fake_rm = FakeVisaRM(
            responses={
                ":READ?": "0.1,1e-6",
            }
        )
        with patch("pyvisa.ResourceManager", return_value=fake_rm):
            smu = KeithleySMU(address="USB0::FAKE::INSTR")
            try:
                smu.smu.command_log.clear()
                current = smu.get_contact_current(0.1, compliance=1e-3, settle=0.0)
                log = smu.smu.get_command_log()

                self.assertAlmostEqual(current, 1e-6)
                self.assertIn(":SOUR:FUNC VOLT", log)
                self.assertIn(':SENS:FUNC "CURR"', log)
                self.assertIn(":OUTP ON", log)
                self.assertIn(":READ?", log)
                self.assertIn(":OUTP OFF", log)
            finally:
                smu.close_session()

    def test_keithley_source_voltage_measure_current_sequence(self):
        from vipsa.hardware.keithley_2450 import KeithleySMU

        fake_rm = FakeVisaRM(
            responses={
                "*OPC?": "1",
                ':TRAC:ACT? "defbuffer1"': "3",
                ':TRAC:DATA? 1, 3, "defbuffer1", SOUR, READ, REL': "0.1,1e-6,0,0.2,2e-6,0.01,0.3,3e-6,0.02",
            }
        )
        with patch("pyvisa.ResourceManager", return_value=fake_rm):
            smu = KeithleySMU(address="USB0::FAKE::INSTR")
            try:
                smu.smu.command_log.clear()
                records = smu.source_voltage_measure_current(
                    voltages=[0.1, 0.2, 0.3],
                    current_compliance=1e-5,
                    delay_s=0.01,
                )
                log = smu.smu.get_command_log()

                self.assertEqual(len(records), 3)
                self.assertIn(":SOUR:LIST:VOLT 0.1,0.2,0.3", log)
                self.assertIn(':SOUR:SWE:VOLT:LIST 1, 0.01, 1, OFF, "defbuffer1"', log)
                self.assertIn(":INIT", log)
                self.assertIn("*OPC?", log)
                self.assertIn(':TRAC:DATA? 1, 3, "defbuffer1", SOUR, READ, REL', log)
            finally:
                smu.close_session()

    def test_keysight_pulse_train_sequence(self):
        from vipsa.hardware.keysight_b2902b import KeysightSMU

        fake_rm = FakeVisaRM(
            responses={
                "*OPC?": "1",
                ":FETC:ARR:SOUR? (@1)": "0.1,0,-0.1",
                ":FETC:ARR:VOLT? (@1)": "0.101,0,-0.099",
                ":FETC:ARR:CURR? (@1)": "1e-6,0,-1e-6",
                ":FETC:ARR:TIME? (@1)": "0,0.001,0.002",
            }
        )
        with patch("pyvisa.ResourceManager", return_value=fake_rm):
            smu = KeysightSMU(address="USB0::FAKE::INSTR")
            try:
                smu.smu.command_log.clear()
                records = smu.run_voltage_pulse_train(
                    voltages=[0.1, 0.0, -0.1],
                    current_compliance=1e-4,
                    pulse_width_s=0.001,
                    acquire_delay_s=0.0005,
                )
                log = smu.smu.get_command_log()

                self.assertEqual(len(records), 3)
                self.assertIn(":SOUR1:FUNC:MODE VOLT", log)
                self.assertIn(":SOUR1:FUNC:SHAP PULS", log)
                self.assertIn(":SOUR1:VOLT:MODE LIST", log)
                self.assertIn(":SOUR1:LIST:VOLT 0.1,0,-0.1", log)
                self.assertIn(":TRIG1:ACQ:DEL 0.0005", log)
                self.assertIn(":INIT (@1)", log)
                self.assertTrue(_contains(log, ":FETC:ARR:SOUR?"))
                self.assertTrue(_contains(log, ":FETC:ARR:VOLT?"))
                self.assertTrue(_contains(log, ":FETC:ARR:CURR?"))
                self.assertTrue(_contains(log, ":FETC:ARR:TIME?"))
            finally:
                smu.close_session()


if __name__ == "__main__":
    unittest.main()
