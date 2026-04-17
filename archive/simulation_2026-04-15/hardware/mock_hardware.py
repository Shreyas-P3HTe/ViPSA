import math
import os
import time

import numpy as np
import pandas as pd


MOCK_DISCOVERED_INSTRUMENTS = {
    "switch": "SIM::707B",
    "keithley": "SIM::Keithley2450",
    "keysight": "SIM::KeysightB2901BL",
}


def _coerce_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_voltage_list(csv_path=None, bare_list=None):
    if bare_list is not None:
        return [float(value) for value in bare_list]

    if csv_path and os.path.exists(csv_path):
        frame = pd.read_csv(csv_path)
        if frame.shape[1] >= 2:
            return frame.iloc[:, 1].astype(float).tolist()
        if frame.shape[1] == 1:
            return frame.iloc[:, 0].astype(float).tolist()

    return [0.0, 0.05, 0.1, 0.05, 0.0, -0.05, -0.1, -0.05, 0.0]


def _record(timestamp, voltage, current, cycle_number=np.nan):
    return {
        "Time(T)": float(timestamp),
        "Voltage (V)": float(voltage),
        "Current (A)": float(current),
        "V_cmd (V)": float(voltage),
        "V_meas (V)": float(voltage),
        "V_error (V)": 0.0,
        "Cycle Number": cycle_number,
    }


class MockSerialConnection:
    def __init__(self, port):
        self.port = port
        self.closed = False

    def close(self):
        self.closed = True


class MockZaberAxis:
    def __init__(self, name, connection):
        self.name = name
        self.position = 0.0
        self.device = type("MockZaberDeviceRef", (), {"connection": connection})()

    def move_relative(self, distance):
        self.position += float(distance)

    def move_absolute(self, position):
        self.position = float(position)


class MockZaberDevice:
    def __init__(self, axis):
        self._axis = axis

    def get_axis(self, _axis_number):
        return self._axis


class MockZaber:
    def __init__(self, port):
        self.port = port
        self.connection = MockSerialConnection(port)
        self.x1 = MockZaberAxis("X", self.connection)
        self.y1 = MockZaberAxis("Y", self.connection)
        self.device_list = [MockZaberDevice(self.y1), MockZaberDevice(self.x1)]

    def get_devices(self):
        return self.x1, self.y1

    def disconnect(self):
        self.connection.close()


class MockStage:
    def __init__(self, port="SIM::COM5", baudrate=115200, multiplier=1.0):
        self.port = port
        self.baudrate = int(baudrate)
        self.scale = float(multiplier)
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.z_pos = 0.0
        self.connected = True
        self.motion_aborted = False

    def flush(self):
        return None

    def is_busy(self):
        return "Idle"

    def change_scale(self, new_scale):
        self.scale = float(new_scale)

    def set_zero(self):
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.z_pos = 0.0

    def go_to_zero(self):
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.z_pos = 0.0

    def get_current_position(self):
        return self.x_pos, self.y_pos, self.z_pos

    def move_x_by(self, steps):
        self.x_pos += float(steps)

    def move_y_by(self, steps):
        self.y_pos += float(steps)

    def move_xy_by(self, steps_x, steps_y):
        self.x_pos += float(steps_x)
        self.y_pos += float(steps_y)

    def move_xy_to(self, x_pos, y_pos):
        self.x_pos = float(x_pos)
        self.y_pos = float(y_pos)

    def move_with_lim_by(self, steps_x, steps_y):
        next_x = self.x_pos + float(steps_x)
        next_y = self.y_pos + float(steps_y)
        if not (-37.5 <= next_x <= 37.5 and -37.5 <= next_y <= 37.5):
            return -1
        self.x_pos = next_x
        self.y_pos = next_y
        return 0

    def move_z_by(self, height, feedrate=450):
        del feedrate
        self.z_pos += float(height)

    def move_z_infinite(self, max_height=300, feedrate=450):
        del feedrate
        self.z_pos += float(max_height)

    def abort_motion(self):
        self.motion_aborted = True

    def reconnect(self):
        self.connected = True
        self.motion_aborted = False

    def speak_grbl(self, gcode):
        return f"mock-stage-received:{gcode}"

    def disconnect(self):
        self.connected = False


class MockLight:
    def __init__(self):
        self.state = "off"
        self.connected = True

    def control_lights(self, command):
        self.state = str(command)

    def disconnect(self):
        self.state = "off"
        self.connected = False


class MockKeithley707B:
    DEFAULT_SLOT = 1
    ROUTE_MAP = {
        "keithley": (("A", 1), ("B", 3)),
        "keysight": (("A", 2), ("B", 4)),
    }

    def __init__(self, address=None, slot=DEFAULT_SLOT):
        self.address = address or MOCK_DISCOVERED_INSTRUMENTS["switch"]
        self.slot = int(slot)
        self._closed_channels = set()

    def build_channel(self, row, column, slot=None):
        slot = self.slot if slot is None else int(slot)
        return f"{slot}{str(row).strip().upper()}{int(column):02d}"

    def _normalize_channel(self, channel):
        if isinstance(channel, dict):
            return self.build_channel(channel["row"], channel["column"], slot=channel.get("slot", self.slot))
        if isinstance(channel, tuple):
            if len(channel) == 2:
                return self.build_channel(channel[0], channel[1])
            if len(channel) == 3:
                return self.build_channel(channel[1], channel[2], slot=channel[0])
        if isinstance(channel, str):
            text = channel.strip()
            if text.lower() in {"all", "allslots"}:
                return "allslots"
            return text.upper()
        return str(channel).upper()

    def _normalize_channel_list(self, channels):
        if isinstance(channels, (list, tuple, set)):
            return [self._normalize_channel(channel) for channel in channels]
        return [self._normalize_channel(channels)]

    def close_channel(self, channel):
        for normalized in self._normalize_channel_list(channel):
            if normalized != "ALLSLOTS":
                self._closed_channels.add(normalized)

    def open_channel(self, channel):
        for normalized in self._normalize_channel_list(channel):
            if normalized == "ALLSLOTS":
                self._closed_channels.clear()
            else:
                self._closed_channels.discard(normalized)

    def open_all(self):
        self._closed_channels.clear()

    def get_closed_channels(self):
        if not self._closed_channels:
            return "nil"
        return ",".join(sorted(self._closed_channels))

    def get_route_channels(self, route_name, slot=None):
        route_key = str(route_name).strip().lower()
        return [
            self.build_channel(row=row, column=column, slot=slot)
            for row, column in self.ROUTE_MAP[route_key]
        ]

    def connect_named_route(self, route_name, slot=None):
        channels = self.get_route_channels(route_name, slot=slot)
        self.open_all()
        self.close_channel(channels)
        return channels

    def connect_keithley_smu(self, slot=None):
        return self.connect_named_route("keithley", slot=slot)

    def connect_keysight_smu(self, slot=None):
        return self.connect_named_route("keysight", slot=slot)

    def reset(self):
        self.open_all()


class MockSMUBase:
    switch_profile = "mock"

    def __init__(self, label, address, switch=None, switch_channel=None, stage=None):
        self.label = label
        self.address = address
        self.switch = switch
        self.switch_channel = switch_channel or self.switch_profile
        self.stage = stage
        self.output_enabled = False
        self.aborted = False
        self.terminal = "REAR"
        self.source_voltage = 0.0
        self.compliance_current = 1e-3
        self._trace_counter = 0
        self.id = f"MOCK,{self.label},0,1.0"
        self.resistance_df = pd.DataFrame(
            {
                "Time(T)": [0.0, 0.01, 0.02, 0.03],
                "Voltage (V)": [0.05, 0.05, -0.05, -0.05],
            }
        )

    def set_stage_reference(self, stage):
        self.stage = stage

    def connect_switch_path(self):
        if self.switch is None:
            return
        route = self.switch_channel if self.switch_channel is not None else self.switch_profile
        route_name = route.lower() if isinstance(route, str) else None
        if route_name in {"keithley", "keysight"} and hasattr(self.switch, "connect_named_route"):
            self.switch.connect_named_route(route_name)
            return
        if hasattr(self.switch, "open_all"):
            self.switch.open_all()
        if hasattr(self.switch, "close_channel"):
            self.switch.close_channel(route)

    def disconnect_switch_path(self):
        if self.switch is None:
            return
        if hasattr(self.switch, "open_all"):
            self.switch.open_all()
            return
        if self.switch_channel is not None and hasattr(self.switch, "open_channel"):
            self.switch.open_channel(self.switch_channel)

    def close_session(self):
        self.abort_measurement()
        self.disconnect_switch_path()

    def get_address(self):
        return self.address

    def reset_device(self):
        self.aborted = False
        self.output_enabled = False
        self.source_voltage = 0.0

    def write(self, command):
        text = str(command).strip()
        upper = text.upper()
        if upper == "*RST":
            self.reset_device()
        elif upper.startswith(":ROUT:TERM"):
            parts = text.split()
            if len(parts) >= 2:
                self.terminal = parts[-1].upper()
        elif upper.startswith(":SOUR:VOLT:LEV"):
            try:
                self.source_voltage = float(text.split()[-1])
            except Exception:
                pass
        elif upper.startswith(":SENS:CURR:PROT"):
            try:
                self.compliance_current = abs(float(text.split()[-1]))
            except Exception:
                pass
        elif upper == ":OUTP ON":
            self.output_enabled = True
        elif upper == ":OUTP OFF":
            self.output_enabled = False
        elif upper == ":ABOR":
            self.aborted = True
            self.output_enabled = False

    def ask(self, command):
        text = str(command).strip().upper()
        if text == "*IDN?":
            return self.id
        if text == ":ROUT:TERM?":
            return self.terminal
        return "0"

    def prepare_contact_probe(self, voltage, compliance):
        self.connect_switch_path()
        self.compliance_current = abs(float(compliance))
        self.source_voltage = float(voltage)
        self.output_enabled = True

    def stop_output(self):
        self.output_enabled = False

    def abort_measurement(self):
        self.aborted = True
        self.output_enabled = False

    def _contact_current(self, voltage):
        stage_z = 0.0
        if self.stage is not None:
            try:
                _x, _y, stage_z = self.stage.get_current_position()
            except Exception:
                stage_z = 0.0

        base_current = 2e-9 if abs(float(voltage)) >= 0.08 else 2.5e-11
        penetration = stage_z - 5.0

        if penetration < 0:
            amplitude = max(1e-12, base_current * 0.02)
        elif penetration <= 1.0:
            amplitude = base_current * (1.0 + 1.5 * penetration)
        else:
            amplitude = base_current * (2.5 + 4.0 * (penetration - 1.0))

        amplitude = min(abs(amplitude), max(self.compliance_current, abs(amplitude)))
        sign = -1.0 if float(voltage) < 0 else 1.0
        return sign * amplitude

    def get_contact_current(self, voltage, compliance=0.1, nplc=1.0, settle=0.02, adr=None):
        del nplc, adr
        self.compliance_current = abs(float(compliance))
        if settle:
            time.sleep(min(float(settle), 0.01))
        return self._contact_current(voltage)

    def get_contact_current_fast(self, voltage, settle=0.02):
        if settle:
            time.sleep(min(float(settle), 0.01))
        return self._contact_current(voltage)

    def _simulate_current(self, voltage, compliance, index):
        base_g = 6e-7 if "Keysight" in self.label else 4.2e-7
        modulation = 1.0 + 0.08 * math.sin(index / 2.0) + 0.12 * math.tanh(float(voltage) * 4.0)
        leakage = 4e-10 * math.sin(index / 3.0)
        current = float(voltage) * base_g * modulation + leakage
        if compliance is not None:
            limit = abs(float(compliance))
            current = max(min(current, limit), -limit)
        return current

    def _emit_progress(self, pending, progress_callback):
        if callable(progress_callback) and pending:
            progress_callback(list(pending))

    def _build_sweep_dataframe(self, voltages, compliance, progress_callback=None, cycle_numbers=None):
        records = []
        pending = []
        start = time.perf_counter()
        for index, voltage in enumerate(voltages):
            current = self._simulate_current(voltage, compliance, index)
            timestamp = time.perf_counter() - start + index * 0.002
            cycle_number = np.nan if cycle_numbers is None else cycle_numbers[index]
            record = _record(timestamp, voltage, current, cycle_number=cycle_number)
            records.append(record)
            pending.append((record["Time(T)"], record["Voltage (V)"], record["Current (A)"]))
            if len(pending) >= 12:
                self._emit_progress(pending, progress_callback)
                pending = []
        self._emit_progress(pending, progress_callback)
        return pd.DataFrame.from_records(records)

    def _build_resistance_dataframe(self, cycles=4):
        rows = []
        for cycle in range(1, cycles + 1):
            voltage = 0.05 if cycle % 2 else -0.05
            current = self._simulate_current(voltage, 2e-5, cycle * 3)
            rows.append(_record(cycle * 0.01, voltage, current, cycle_number=cycle))
        return pd.DataFrame.from_records(rows)

    def list_IV_sweep_split(
        self,
        csv_path,
        pos_compliance,
        neg_compliance,
        SMU_range=None,
        delay=None,
        acq_delay=None,
        adr=None,
        pos_channel=None,
        neg_channel=None,
        wait_time=None,
        progress_callback=None,
        include_read_probe=True,
        read_probe_mode="between_segments",
    ):
        del SMU_range, delay, acq_delay, adr, pos_channel, neg_channel, wait_time, read_probe_mode
        self.connect_switch_path()
        voltages = _load_voltage_list(csv_path=csv_path)
        compliance = max(abs(float(pos_compliance)), abs(float(neg_compliance)))
        cycle_numbers = [1 + (index // max(1, len(voltages))) for index in range(len(voltages))]
        sweep = self._build_sweep_dataframe(voltages, compliance, progress_callback=progress_callback, cycle_numbers=cycle_numbers)
        resistance = self._build_resistance_dataframe() if include_read_probe else pd.DataFrame()
        self.stop_output()
        self.disconnect_switch_path()
        return sweep, resistance

    def list_IV_sweep_split_4(
        self,
        csv_path,
        compliance_pf,
        compliance_pb,
        compliance_nf,
        compliance_nb,
        delay=None,
        nplc=0.01,
        wait_time=0.0,
        progress_callback=None,
        include_read_probe=True,
        read_probe_mode="between_segments",
        current_autorange=False,
    ):
        del delay, nplc, wait_time, read_probe_mode, current_autorange
        self.connect_switch_path()
        voltages = _load_voltage_list(csv_path=csv_path)
        compliance = max(abs(float(compliance_pf)), abs(float(compliance_pb)), abs(float(compliance_nf)), abs(float(compliance_nb)))
        cycle_numbers = [1 for _ in voltages]
        sweep = self._build_sweep_dataframe(voltages, compliance, progress_callback=progress_callback, cycle_numbers=cycle_numbers)
        resistance = self._build_resistance_dataframe(cycles=6) if include_read_probe else pd.DataFrame()
        self.stop_output()
        self.disconnect_switch_path()
        return sweep, resistance

    def pulsed_measurement(
        self,
        csv_path,
        current_compliance,
        set_width=0.01,
        bare_list=None,
        set_acquire_delay=None,
        adr=None,
        current_autorange=False,
    ):
        del set_width, set_acquire_delay, adr, current_autorange
        self.connect_switch_path()
        voltages = _load_voltage_list(csv_path=csv_path, bare_list=bare_list)
        records = self._build_sweep_dataframe(
            voltages,
            abs(float(current_compliance)),
            progress_callback=None,
            cycle_numbers=[index + 1 for index in range(len(voltages))],
        )
        self.stop_output()
        self.disconnect_switch_path()
        return records


class MockKeithleySMU(MockSMUBase):
    switch_profile = "keithley"

    def __init__(self, address=None, switch=None, switch_channel="keithley", stage=None):
        super().__init__(
            label="Keithley2450",
            address=address or MOCK_DISCOVERED_INSTRUMENTS["keithley"],
            switch=switch,
            switch_channel=switch_channel,
            stage=stage,
        )


class MockKeysightSMU(MockSMUBase):
    switch_profile = "keysight"

    def __init__(self, address=None, switch=None, switch_channel="keysight", stage=None):
        super().__init__(
            label="KeysightB2901BL",
            address=address or MOCK_DISCOVERED_INSTRUMENTS["keysight"],
            switch=switch,
            switch_channel=switch_channel,
            stage=stage,
        )


def build_mock_smu(label, switch=None, stage=None, address=None):
    text = str(label).strip().lower()
    if "keithley" in text or "2450" in text:
        return MockKeithleySMU(address=address, switch=switch, stage=stage)
    if "keysight" in text or "b290" in text:
        return MockKeysightSMU(address=address, switch=switch, stage=stage)
    raise ValueError(f"Unknown mock SMU label: {label}")


def build_mock_setup(arduino_port="SIM::COM5", arduino_baud=115200, arduino_scale=1.0, zaber_port="SIM::COM7"):
    stage = MockStage(port=arduino_port, baudrate=arduino_baud, multiplier=arduino_scale)
    zaber = MockZaber(zaber_port)
    zaber_x, zaber_y = zaber.get_devices()
    light = MockLight()
    switch = MockKeithley707B()
    return {
        "stage": stage,
        "Zaber": zaber,
        "zaber_x": zaber_x,
        "zaber_y": zaber_y,
        "top_light": light,
        "switch": switch,
        "discovered_instruments": dict(MOCK_DISCOVERED_INSTRUMENTS),
    }


__all__ = [
    "MOCK_DISCOVERED_INSTRUMENTS",
    "MockKeithley707B",
    "MockKeithleySMU",
    "MockKeysightSMU",
    "MockLight",
    "MockStage",
    "MockZaber",
    "build_mock_setup",
    "build_mock_smu",
    "_coerce_bool",
]
