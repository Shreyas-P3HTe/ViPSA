from __future__ import annotations

import os
import sys
from typing import Any

import pyvisa

PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from vipsa.hardware.keithley_2450 import KeithleySMU
from vipsa.hardware.keysight_b2902b import KeysightSMU
from vipsa.hardware.keithley_707b import Keithley707B


class TraceSession:
    def __init__(self, session: Any, label: str) -> None:
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_label", label)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_session", "_label"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._session, name, value)

    def write(self, command: str) -> Any:
        print(f"[{self._label}] >> {command}")
        return self._session.write(command)

    def query(self, command: str) -> str:
        print(f"[{self._label}] ?? {command}")
        response = self._session.query(command)
        print(f"[{self._label}] << {response}")
        return response

    def read(self) -> str:
        response = self._session.read()
        print(f"[{self._label}] << {response}")
        return response

    def close(self) -> Any:
        return self._session.close()


def _wrap_session(session: Any, label: str) -> TraceSession:
    if isinstance(session, TraceSession):
        return session
    return TraceSession(session, label)


def _attach_driver_trace(driver: Any, label: str) -> Any:
    if getattr(driver, "smu", None) is not None:
        driver.smu = _wrap_session(driver.smu, label)

    original_open = driver._open_resource

    def traced_open(adr: str | None = None, timeout: int = 10000):
        return _wrap_session(original_open(adr=adr, timeout=timeout), label)

    driver._open_resource = traced_open
    return driver


def _attach_switch_trace(switch: Keithley707B) -> Keithley707B:
    original_open = switch._open_device

    def traced_open():
        dev = original_open()
        wrapped = _wrap_session(dev, "707B")
        switch._dev = wrapped
        return wrapped

    switch._open_device = traced_open
    return switch


def _safe_stop_output(driver: Any | None) -> None:
    if driver is None:
        return
    try:
        driver.stop_output()
    except Exception:
        pass


def _safe_close(device: Any | None) -> None:
    if device is None:
        return
    try:
        device.close_session()
    except Exception:
        try:
            device.close()
        except Exception:
            pass


def _print_records(records: list[dict[str, Any]]) -> None:
    if not records:
        print("No records.")
        return
    for record in records:
        print(
            "t={:.6g} Vcmd={} Vmeas={} Imeas={}".format(
                float(record.get("Time(T)", 0.0)),
                record.get("V_cmd (V)"),
                record.get("V_meas (V)"),
                record.get("I_meas (A)", record.get("Current (A)")),
            )
        )


def _prompt_text(label: str, default: str = "") -> str:
    text = input(f"{label} [{default}]: ").strip()
    return text or default


def _prompt_float(label: str, default: float) -> float:
    return float(_prompt_text(label, str(default)))


def _prompt_voltage_list(label: str, default: str) -> list[float]:
    raw = _prompt_text(label, default)
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


class BenchConfig:
    def __init__(self) -> None:
        self.keithley_addr = os.getenv("VIPSA_KEITHLEY_ADDR", "")
        self.keysight_addr = os.getenv("VIPSA_KEYSIGHT_ADDR", "")
        self.switch_addr = os.getenv("VIPSA_707B_ADDR", "")

    def _resolve_address(self, label: str, current: str) -> str:
        return _prompt_text(label, current)

    def make_switch(self) -> Keithley707B:
        self.switch_addr = self._resolve_address("707B VISA address", self.switch_addr)
        kwargs = {"address": self.switch_addr} if self.switch_addr else {"device_no": 0}
        return _attach_switch_trace(Keithley707B(**kwargs))

    def make_keithley(self, switch: Keithley707B | None = None) -> KeithleySMU:
        self.keithley_addr = self._resolve_address("Keithley VISA address", self.keithley_addr)
        kwargs = {"address": self.keithley_addr} if self.keithley_addr else {"device_no": 0}
        kwargs.update({"switch": switch, "switch_channel": "keithley", "connect_switch": False})
        return _attach_driver_trace(KeithleySMU(**kwargs), "Keithley")

    def make_keysight(self, switch: Keithley707B | None = None) -> KeysightSMU:
        self.keysight_addr = self._resolve_address("Keysight VISA address", self.keysight_addr)
        kwargs = {"address": self.keysight_addr} if self.keysight_addr else {"device_no": 0}
        kwargs.update({"switch": switch, "switch_channel": "keysight", "connect_switch": False})
        return _attach_driver_trace(KeysightSMU(**kwargs), "Keysight")


def list_visa_resources() -> None:
    try:
        resources = tuple(pyvisa.ResourceManager().list_resources())
    except Exception as exc:
        print(f"ERROR: {exc}")
        return
    if not resources:
        print("No VISA resources found.")
        return
    for index, resource in enumerate(resources, start=1):
        print(f"{index}. {resource}")


def identify_keithley(config: BenchConfig) -> None:
    keithley = None
    try:
        keithley = config.make_keithley()
        print(keithley.identify())
    finally:
        _safe_stop_output(keithley)
        _safe_close(keithley)


def identify_keysight(config: BenchConfig) -> None:
    keysight = None
    try:
        keysight = config.make_keysight()
        print(keysight.identify())
    finally:
        _safe_stop_output(keysight)
        _safe_close(keysight)


def identify_707b(config: BenchConfig) -> None:
    switch = None
    try:
        switch = config.make_switch()
        for expression in ("localnode.model", "localnode.serialno", "_VERSION"):
            try:
                value = switch.query_tsp(expression)
                print(f"{expression} = {value}")
            except Exception:
                continue
        print(f"closed_channels = {switch.get_closed_channels()}")
    finally:
        _safe_close(switch)


def open_all_707b(config: BenchConfig) -> None:
    switch = None
    try:
        switch = config.make_switch()
        switch.open_all()
        print(f"closed_channels = {switch.get_closed_channels()}")
    finally:
        _safe_close(switch)


def route_switch(config: BenchConfig, route_name: str) -> None:
    switch = None
    driver = None
    try:
        switch = config.make_switch()
        if route_name == "keithley":
            driver = config.make_keithley(switch=switch)
        else:
            driver = config.make_keysight(switch=switch)
        _safe_stop_output(driver)
        switch.open_all()
        channels = switch.connect_named_route(route_name)
        print(f"route={route_name} channels={channels}")
        print(f"closed_channels = {switch.get_closed_channels()}")
    finally:
        _safe_stop_output(driver)
        _safe_close(driver)
        _safe_close(switch)


def keithley_contact_test(config: BenchConfig) -> None:
    switch = None
    keithley = None
    try:
        switch = config.make_switch()
        keithley = config.make_keithley(switch=switch)
        _safe_stop_output(keithley)
        switch.open_all()
        switch.connect_named_route("keithley")
        voltage = _prompt_float("Contact test voltage (V)", 0.1)
        compliance = _prompt_float("Contact test compliance (A)", 0.001)
        current = keithley.get_contact_current(voltage=voltage, compliance=compliance, settle=0.0)
        print(f"measured_current = {current} A")
    finally:
        _safe_stop_output(keithley)
        _safe_close(keithley)
        _safe_close(switch)


def keysight_contact_test(config: BenchConfig) -> None:
    switch = None
    keysight = None
    try:
        switch = config.make_switch()
        keysight = config.make_keysight(switch=switch)
        _safe_stop_output(keysight)
        switch.open_all()
        switch.connect_named_route("keysight")
        voltage = _prompt_float("Contact test voltage (V)", 0.1)
        compliance = _prompt_float("Contact test compliance (A)", 0.001)
        current = keysight.get_contact_current(voltage=voltage, compliance=compliance)
        print(f"measured_current = {current} A")
    finally:
        _safe_stop_output(keysight)
        _safe_close(keysight)
        _safe_close(switch)


def keithley_tiny_iv(config: BenchConfig) -> None:
    switch = None
    keithley = None
    try:
        switch = config.make_switch()
        keithley = config.make_keithley(switch=switch)
        _safe_stop_output(keithley)
        switch.open_all()
        switch.connect_named_route("keithley")
        voltages = _prompt_voltage_list("Keithley tiny IV voltages", "0,0.1,0.2,0.1,0")
        compliance = _prompt_float("Keithley compliance (A)", 0.001)
        delay_s = _prompt_float("Keithley delay per point (s)", 0.01)
        records = keithley.source_voltage_measure_current(
            voltages=voltages,
            current_compliance=compliance,
            delay_s=delay_s,
        )
        _print_records(records)
    finally:
        _safe_stop_output(keithley)
        _safe_close(keithley)
        _safe_close(switch)


def keysight_pulse_train(config: BenchConfig) -> None:
    switch = None
    keysight = None
    try:
        switch = config.make_switch()
        keysight = config.make_keysight(switch=switch)
        _safe_stop_output(keysight)
        switch.open_all()
        switch.connect_named_route("keysight")
        voltages = _prompt_voltage_list("Keysight pulse voltages", "0,1,0,-1,0")
        compliance = _prompt_float("Keysight compliance (A)", 0.001)
        pulse_width = _prompt_float("Pulse width (s)", 0.001)
        acq_delay = _prompt_float("Acquire delay (s)", pulse_width / 2)
        records = keysight.run_voltage_pulse_train(
            voltages=voltages,
            current_compliance=compliance,
            pulse_width_s=pulse_width,
            acquire_delay_s=acq_delay,
        )
        _print_records(records)
    finally:
        _safe_stop_output(keysight)
        _safe_close(keysight)
        _safe_close(switch)


def main() -> None:
    config = BenchConfig()
    actions = {
        "1": ("List VISA resources", lambda: list_visa_resources()),
        "2": ("Identify Keithley", lambda: identify_keithley(config)),
        "3": ("Identify Keysight", lambda: identify_keysight(config)),
        "4": ("Identify 707B", lambda: identify_707b(config)),
        "5": ("707B open_all", lambda: open_all_707b(config)),
        "6": ("Route Keithley", lambda: route_switch(config, "keithley")),
        "7": ("Route Keysight", lambda: route_switch(config, "keysight")),
        "8": ("Keithley contact current test", lambda: keithley_contact_test(config)),
        "9": ("Keysight contact current test", lambda: keysight_contact_test(config)),
        "10": ("Keithley tiny IV sweep", lambda: keithley_tiny_iv(config)),
        "11": ("Keysight pulse train", lambda: keysight_pulse_train(config)),
        "12": ("Exit", None),
    }

    while True:
        print()
        for key, (label, _) in actions.items():
            print(f"{key}. {label}")
        choice = input("Select: ").strip()
        if choice == "12":
            break
        action = actions.get(choice)
        if action is None:
            print("Invalid selection.")
            continue
        try:
            action[1]()
        except Exception as exc:
            print(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
