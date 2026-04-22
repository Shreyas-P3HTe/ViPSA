"""Pure PyVISA + native SCPI wrapper for the Keithley 2450 SourceMeter."""

# TODO: Resolve remaining legacy compatibility gaps: keep the old SMU/SCPI
# method surface stable while verifying command sequences on a real 2450.

from __future__ import annotations

from typing import Any, Iterable, Sequence
import time

import pyvisa


class _NullVisaSession:
	"""No-op VISA session used only when PyVISA is unavailable in test imports."""

	def __init__(self, address: str) -> None:
		self.address = address
		self.timeout = None
		self.read_termination = "\n"
		self.write_termination = "\n"
		self.writes: list[str] = []

	def write(self, command: str) -> None:
		self.writes.append(command)

	def query(self, command: str) -> str:
		self.writes.append(command)
		if command == "*IDN?":
			return "NO_PYVISA,KEITHLEY2450,0,0"
		if command == "*OPC?":
			return "1"
		if command == ":READ?":
			return "0,0"
		if command.startswith(":TRAC:ACT?"):
			return "1"
		if command.startswith(":TRAC:DATA?"):
			return "0,0,0"
		if command.startswith(":SYST:ERR?"):
			return "0,No error"
		return "0"

	def close(self) -> None:
		return None


class _NullResourceManager:
	"""No-op resource manager used when tests stub ``pyvisa`` without a backend."""

	def list_resources(self) -> tuple[str, ...]:
		return ("USB0::FAKE::INSTR",)

	def open_resource(self, address: str) -> _NullVisaSession:
		return _NullVisaSession(address)


def _resource_manager():
	"""Return a real ResourceManager when available, else a no-op test fallback."""
	factory = getattr(pyvisa, "ResourceManager", None)
	if callable(factory):
		return factory()
	return _NullResourceManager()


def _record(
	timestamp: float,
	v_cmd: float | None = None,
	i_cmd: float | None = None,
	v_meas: float | None = None,
	i_meas: float | None = None,
	cycle_number: float | None = None,
) -> dict[str, float | None]:
	"""Build one normalized measurement record."""
	v_error = None
	i_error = None

	if v_cmd is not None and v_meas is not None:
		v_error = float(v_meas) - float(v_cmd)

	if i_cmd is not None and i_meas is not None:
		i_error = float(i_meas) - float(i_cmd)

	return {
		"Time(T)": float(timestamp),
		"Voltage (V)": float(v_cmd) if v_cmd is not None else v_meas,
		"Current (A)": float(i_meas) if i_meas is not None else i_cmd,
		"V_cmd (V)": float(v_cmd) if v_cmd is not None else None,
		"I_cmd (A)": float(i_cmd) if i_cmd is not None else None,
		"V_meas (V)": float(v_meas) if v_meas is not None else None,
		"I_meas (A)": float(i_meas) if i_meas is not None else None,
		"V_error (V)": v_error,
		"I_error (A)": i_error,
		"Cycle Number": cycle_number,
	}


class Keithley2450:
	"""Native SCPI driver for the Keithley 2450.

	The class keeps a persistent VISA session and provides thin measurement
	primitives. Higher-level voltage-list splitting and GUI compatibility logic
	should live in ``Source_Measure_Unit.py``.
	"""

	instrument_family = "keithley_2450"
	supports_native_pulse = False
	supports_dual_channel = False

	def __init__(
		self,
		device_no: int = 0,
		address: str | None = None,
		switch: Any | None = None,
		switch_channel: Any | None = None,
		connect_switch: bool = False,
	) -> None:
		"""Initialize the driver and open the main VISA session."""
		self.rm = _resource_manager()
		address_list = list(self.rm.list_resources())
		self.address = address if address is not None else address_list[device_no]
		self.switch = switch
		self.switch_channel = switch_channel
		self.switch_profile = "keithley"
		self.smu = self._open_resource(self.address, timeout=10000)

		if connect_switch and self.switch is not None:
			self.connect_switch_path()

		self.initialize()

	def _open_resource(self, adr: str | None = None, timeout: int = 10000):
		"""Open a VISA resource and apply common session settings."""
		target = self.address if adr is None else adr
		smu = self.rm.open_resource(target)
		smu.read_termination = "\n"
		smu.write_termination = "\n"
		smu.timeout = timeout
		return smu

	def _get_session(self, adr: str | None = None, timeout: int = 10000):
		"""Return the main session or open a short-lived one for another address."""
		target = self.address if adr is None else adr
		if target == self.address:
			self.smu.timeout = timeout
			return self.smu, False
		return self._open_resource(target, timeout=timeout), True

	def _close_temp(self, smu, temporary: bool) -> None:
		"""Close a temporary VISA session."""
		if temporary and smu is not None:
			smu.close()

	def _normalize_function(self, function: str) -> str:
		"""Normalize function tokens such as ``VOLT`` and ``CURR``."""
		text = str(function).strip().upper()
		if text.startswith("V"):
			return "VOLT"
		if text.startswith("I") or text.startswith("C"):
			return "CURR"
		raise ValueError(f"Unsupported function '{function}'.")

	def _read_measurement_pair(self, smu) -> tuple[float | None, float | None]:
		"""Read one voltage/current pair using the active configuration."""
		response = smu.query(":READ?").strip()
		parts = [part.strip() for part in response.split(",") if part.strip()]
		if len(parts) >= 2:
			return float(parts[0]), float(parts[1])
		if len(parts) == 1:
			return None, float(parts[0])
		return None, None

	def _write_voltage_source_list(self, smu, voltages: Sequence[float]) -> None:
		"""Write a 2450 voltage source list, chunking around SCPI list limits."""
		if len(voltages) > 2500:
			raise ValueError("The Keithley 2450 source-list limit is 2500 points.")

		for start in range(0, len(voltages), 100):
			chunk = voltages[start:start + 100]
			payload = ",".join(f"{float(value):.12g}" for value in chunk)
			command = ":SOUR:LIST:VOLT" if start == 0 else ":SOUR:LIST:VOLT:APP"
			smu.write(f"{command} {payload}")

	def _read_trace_records(
		self,
		smu,
		voltages: Sequence[float],
		delay_s: float,
		buffer_name: str = "defbuffer1",
	) -> list[dict[str, float | None]]:
		"""Read source/current/time triples from a 2450 trace buffer."""
		try:
			raw_count = smu.query(f':TRAC:ACT? "{buffer_name}"').strip()
			point_count = int(float(raw_count))
		except Exception:
			point_count = len(voltages)

		point_count = max(0, min(point_count, len(voltages)))
		if point_count <= 0:
			point_count = len(voltages)

		raw = smu.query(
			f':TRAC:DATA? 1, {point_count}, "{buffer_name}", SOUR, READ, REL'
		).strip()
		values = [part.strip() for part in raw.split(",") if part.strip()]
		records: list[dict[str, float | None]] = []

		for index in range(0, len(values), 3):
			group = values[index:index + 3]
			if len(group) < 3:
				continue
			try:
				v_meas = float(group[0])
				i_meas = float(group[1])
				timestamp = float(group[2])
			except ValueError:
				continue
			voltage_index = len(records)
			v_cmd = voltages[voltage_index] if voltage_index < len(voltages) else v_meas
			records.append(
				_record(
					timestamp=timestamp,
					v_cmd=v_cmd,
					v_meas=v_meas,
					i_meas=i_meas,
					cycle_number=float(voltage_index),
				)
			)

		if records:
			return records

		return [
			_record(
				timestamp=float(index) * delay_s,
				v_cmd=float(voltage),
				i_meas=None,
				cycle_number=float(index),
			)
			for index, voltage in enumerate(voltages)
		]

	def _set_nplc(self, function: str, nplc: float | None) -> None:
		"""Apply an NPLC value when one is supplied."""
		if nplc is None:
			return
		func = self._normalize_function(function)
		self.write(f":SENS:{func}:NPLC {float(nplc):.12g}")

	def initialize(self) -> "Keithley2450":
		"""Apply a conservative default state suitable for lab automation."""
		self.write("*RST")
		self.write("*CLS")
		self.write(":ROUT:TERM REAR")
		self.write(":SOUR:FUNC VOLT")
		self.write(':SENS:FUNC "CURR"')
		self.write(":SENS:CURR:RANG:AUTO ON")
		self.write(":OUTP OFF")
		return self

	def write(self, command: str, adr: str | None = None) -> None:
		"""Send one SCPI command."""
		smu, temporary = self._get_session(adr=adr, timeout=10000)
		try:
			smu.write(command)
		finally:
			self._close_temp(smu, temporary)

	def query(self, command: str, adr: str | None = None) -> str:
		"""Send one SCPI query and return its stripped response."""
		smu, temporary = self._get_session(adr=adr, timeout=10000)
		try:
			return smu.query(command).strip()
		finally:
			self._close_temp(smu, temporary)

	def ask(self, command: str, adr: str | None = None) -> str:
		"""Backward-compatible alias for ``query``."""
		return self.query(command, adr=adr)

	def identify(self) -> str:
		"""Return the instrument ID string."""
		return self.query("*IDN?")

	def get_address(self) -> str | None:
		"""Return the configured VISA resource address for legacy callers."""
		return self.address

	def connect(
		self,
		address: str | None = None,
		device_no: int = 0,
	) -> "Keithley2450":
		"""Open or reopen the persistent Keithley VISA session."""
		if address is not None:
			self.address = address
		elif self.smu is None:
			addresses = list(self.rm.list_resources())
			self.address = addresses[device_no]

		if self.smu is None:
			self.smu = self._open_resource(self.address, timeout=10000)
		return self

	def disconnect(self) -> None:
		"""Close the active Keithley session using the legacy lifecycle name."""
		self.close_session()

	def clear(self) -> "Keithley2450":
		"""Clear the Keithley status and error queues."""
		self.write("*CLS")
		return self

	def reset(self) -> "Keithley2450":
		"""Reset the Keithley and reapply the ViPSA default setup."""
		self.write("*RST")
		self.clear()
		return self.initialize()

	def set_source_function(self, function: str) -> None:
		"""Select voltage or current source mode."""
		func = self._normalize_function(function)
		self.write(f":SOUR:FUNC {func}")

	def set_sense_function(self, function: str | Sequence[str]) -> None:
		"""Enable one or more sense functions.

		The Keithley 2450 can be configured with a single primary sense function.
		When multiple functions are supplied, the first entry is used.
		"""
		if isinstance(function, (list, tuple)):
			selected = function[0]
		else:
			selected = function
		func = self._normalize_function(str(selected))
		self.write(f':SENS:FUNC "{func}"')

	def set_source_range(
		self,
		function: str,
		value: float | None = None,
		auto: bool = True,
	) -> None:
		"""Configure the source range or source autorange."""
		func = self._normalize_function(function)
		if auto:
			self.write(f":SOUR:{func}:RANG:AUTO ON")
			return
		if value is None:
			raise ValueError("A source range value is required when auto is False.")
		self.write(f":SOUR:{func}:RANG:AUTO OFF")
		self.write(f":SOUR:{func}:RANG {float(value):.12g}")

	def set_sense_range(
		self,
		function: str,
		value: float | None = None,
		auto: bool = True,
	) -> None:
		"""Configure the measurement range or measurement autorange."""
		func = self._normalize_function(function)
		if auto:
			self.write(f":SENS:{func}:RANG:AUTO ON")
			return
		if value is None:
			raise ValueError("A sense range value is required when auto is False.")
		self.write(f":SENS:{func}:RANG:AUTO OFF")
		self.write(f":SENS:{func}:RANG {float(value):.12g}")

	def set_protection(self, function: str, value: float) -> None:
		"""Configure current or voltage protection / compliance."""
		func = self._normalize_function(function)
		numeric = float(value)
		if func == "VOLT":
			self.write(f":SENS:CURR:PROT {numeric:.12g}")
			self.write(f":SOUR:VOLT:ILIM {numeric:.12g}")
			return
		self.write(f":SENS:VOLT:PROT {numeric:.12g}")
		self.write(f":SOUR:CURR:VLIM {numeric:.12g}")

	def set_source_level(self, function: str, value: float) -> None:
		"""Set the immediate source level for voltage or current."""
		func = self._normalize_function(function)
		self.write(f":SOUR:{func}:LEV {float(value):.12g}")

	def set_output(self, enabled: bool) -> None:
		"""Turn the output on or off."""
		self.write(":OUTP ON" if enabled else ":OUTP OFF")

	def configure_measurement_format(self, functions: str | Sequence[str]) -> None:
		"""Configure the response fields for immediate readback.

		The 2450 uses ``:FORM:ELEM`` rather than the Keysight ``:FORM:ELEM:SENS``
		variant. Typical values are ``CURR`` or ``VOLT,CURR``.
		"""
		if isinstance(functions, str):
			payload = functions
		else:
			payload = ",".join(str(item).strip().upper() for item in functions)
		self.write(f":FORM:ELEM {payload}")

	def query_reading(self) -> str:
		"""Query one reading from the active source-measure setup."""
		return self.query(":READ?")

	def get_error(self, SMU: Any | None = None) -> list[str]:
		"""Return all currently queued Keithley instrument errors."""
		dev = self.smu if SMU is None else SMU
		errors: list[str] = []
		while True:
			msg = dev.query(":SYST:ERR?").strip()
			if msg.startswith("+0") or msg.startswith("0"):
				break
			errors.append(msg)
		return errors

	def read_errors(self) -> list[str]:
		"""Drain the instrument error queue using the active session."""
		return self.get_error()

	def connect_switch_path(self) -> None:
		"""Connect the assigned switch path if a switch wrapper is available."""
		if self.switch is None:
			return

		route = self.switch_channel if self.switch_channel is not None else self.switch_profile
		route_name = route.lower() if isinstance(route, str) else None

		if hasattr(self.switch, "connect_named_route") and route_name in {"keithley", "keysight"}:
			self.switch.connect_named_route(route_name)
			return

		if hasattr(self.switch, "open_all"):
			self.switch.open_all()

		if hasattr(self.switch, "close_channel"):
			self.switch.close_channel(route)

	def disconnect_switch_path(self) -> None:
		"""Open the assigned switch path if a switch wrapper is available."""
		if self.switch is None:
			return

		if hasattr(self.switch, "open_all"):
			self.switch.open_all()
			return

		if self.switch_channel is not None and hasattr(self.switch, "open_channel"):
			self.switch.open_channel(self.switch_channel)

	def close_session(self) -> None:
		"""Abort output, disconnect the switch, and close the VISA session."""
		try:
			self.abort_measurement()
		except Exception:
			pass

		try:
			self.disconnect_switch_path()
		except Exception:
			pass

		try:
			if self.smu is not None:
				self.smu.close()
		finally:
			self.smu = None

	def reset_device(self) -> None:
		"""Reset and reinitialize the driver."""
		self.initialize()

	def stop_output(self) -> None:
		"""Turn the output off."""
		self.set_output(False)

	def abort_measurement(self) -> None:
		"""Abort any active operation and leave the output off."""
		try:
			self.write(":ABOR")
		except Exception:
			pass
		try:
			self.set_output(False)
		except Exception:
			pass

	def prepare_contact_probe(self, voltage: float, compliance: float) -> None:
		"""Prepare a steady voltage bias for contact-current probing."""
		self.connect_switch_path()
		self.initialize()
		self.set_source_function("VOLT")
		self.set_sense_function("CURR")
		self.set_protection("VOLT", compliance)
		self.set_source_level("VOLT", voltage)
		self.set_output(True)

	def get_contact_current(
		self,
		voltage: float,
		compliance: float = 0.1,
		nplc: float = 1.0,
		settle: float = 0.02,
		adr: str | None = None,
	) -> float:
		"""Apply one voltage bias and return the absolute measured current."""
		records = self.hold_voltage_measure_current(
			voltage=voltage,
			current_compliance=compliance,
			settle_s=settle,
			nplc=nplc,
			read_count=1,
			reset=True,
			adr=adr,
		)
		if not records:
			return 0.0
		value = records[0].get("Current (A)")
		return abs(float(value)) if value is not None else 0.0

	def get_contact_current_fast(
		self,
		voltage: float,
		settle: float = 0.02,
	) -> float:
		"""Read absolute current quickly assuming the instrument is configured."""
		self.set_source_level("VOLT", voltage)
		self.set_output(True)
		if settle > 0:
			time.sleep(settle)
		_, current = self._read_measurement_pair(self.smu)
		return abs(float(current)) if current is not None else 0.0

	def read_current_at_voltage(
		self,
		voltage: float,
		current_compliance: float = 0.1,
		delay_s: float = 0.02,
		**kwargs: Any,
	) -> float:
		"""Legacy helper that returns measured current at one voltage point."""
		records = self.hold_voltage_measure_current(
			voltage=voltage,
			current_compliance=current_compliance,
			settle_s=delay_s,
			read_count=1,
			reset=bool(kwargs.get("reset", True)),
			adr=kwargs.get("adr"),
		)
		if not records:
			return 0.0
		value = records[0].get("Current (A)")
		return float(value) if value is not None else 0.0

	def measure_resistance(
		self,
		voltage: float = 0.1,
		current_compliance: float = 0.1,
		delay_s: float = 0.02,
		**kwargs: Any,
	) -> float:
		"""Legacy helper that estimates resistance from a one-point IV reading."""
		current = self.read_current_at_voltage(
			voltage=voltage,
			current_compliance=current_compliance,
			delay_s=delay_s,
			**kwargs,
		)
		if current == 0:
			return float("inf")
		return float(voltage) / current

	def pulsed_measurement(
		self,
		csv_path: str | None,
		current_compliance: float,
		set_width: float = 0.01,
		bare_list: Sequence[float] | None = None,
		set_acquire_delay: float | None = None,
		adr: str | None = None,
		current_autorange: bool = False,
	) -> list[dict[str, float | None]]:
		"""Legacy wrapper for a Keithley voltage pulse/list measurement."""
		_ = csv_path
		voltages = [] if bare_list is None else [float(value) for value in bare_list]
		return self.run_voltage_pulse_train(
			voltages=voltages,
			current_compliance=current_compliance,
			pulse_width_s=set_width,
			acquire_delay_s=set_acquire_delay,
			adr=adr,
			current_autorange=current_autorange,
		)

	def run_read_probe(self, *args: Any, **kwargs: Any) -> None:
		"""Legacy placeholder for handler-level read-probe orchestration."""
		return None

	def identify_linear_segments(self, *args: Any, **kwargs: Any) -> None:
		"""Legacy placeholder for handler-level linear segment detection."""
		return None

	def split_by_polarity(self, *args: Any, **kwargs: Any) -> None:
		"""Legacy placeholder for handler-level polarity segmentation."""
		return None

	def split_sweep_by_4(self, *args: Any, **kwargs: Any) -> None:
		"""Legacy placeholder for handler-level four-way sweep segmentation."""
		return None

	def list_IV_sweep_split_4(self, *args: Any, **kwargs: Any) -> None:
		"""Legacy placeholder for handler-level split IV sweep execution."""
		return None

	def run_linear_segment(self, *args: Any, **kwargs: Any) -> None:
		"""Legacy placeholder for handler-level linear segment execution."""
		return None

	def list_IV_sweep_split(self, *args: Any, **kwargs: Any) -> None:
		"""Legacy placeholder for handler-level split IV sweep execution."""
		return None

	def hold_voltage_measure_current(
		self,
		voltage: float,
		current_compliance: float,
		settle_s: float = 0.0,
		current_range: float | None = None,
		voltage_range: float | None = None,
		nplc: float | None = None,
		read_count: int = 1,
		reset: bool = True,
		adr: str | None = None,
	) -> list[dict[str, float | None]]:
		"""Apply a constant voltage bias and measure current one or more times."""
		smu, temporary = self._get_session(adr=adr, timeout=30000)
		try:
			if reset:
				smu.write("*RST")
				smu.write("*CLS")
			smu.write(":ROUT:TERM REAR")
			smu.write(":SOUR:FUNC VOLT")
			smu.write(':SENS:FUNC "CURR"')
			smu.write(f":SENS:CURR:PROT {float(current_compliance):.12g}")
			smu.write(f":SOUR:VOLT:ILIM {float(current_compliance):.12g}")

			if voltage_range is not None:
				smu.write(":SOUR:VOLT:RANG:AUTO OFF")
				smu.write(f":SOUR:VOLT:RANG {float(voltage_range):.12g}")
			else:
				smu.write(":SOUR:VOLT:RANG:AUTO ON")

			if current_range is not None:
				smu.write(":SENS:CURR:RANG:AUTO OFF")
				smu.write(f":SENS:CURR:RANG {float(current_range):.12g}")
			else:
				smu.write(":SENS:CURR:RANG:AUTO ON")

			if nplc is not None:
				smu.write(f":SENS:CURR:NPLC {float(nplc):.12g}")

			smu.write(":FORM:ELEM VOLT,CURR")
			smu.write(f":SOUR:VOLT:LEV {float(voltage):.12g}")
			smu.write(":OUTP ON")

			if settle_s > 0:
				time.sleep(settle_s)

			t0 = time.perf_counter()
			records: list[dict[str, float | None]] = []
			for index in range(max(1, int(read_count))):
				v_meas, i_meas = self._read_measurement_pair(smu)
				records.append(
					_record(
						timestamp=time.perf_counter() - t0,
						v_cmd=voltage,
						v_meas=v_meas,
						i_meas=i_meas,
						cycle_number=float(index),
					)
				)

			smu.write(":OUTP OFF")
			return records
		finally:
			self._close_temp(smu, temporary)

	def hold_current_measure_voltage(
		self,
		current: float,
		voltage_compliance: float,
		settle_s: float = 0.0,
		current_range: float | None = None,
		voltage_range: float | None = None,
		nplc: float | None = None,
		read_count: int = 1,
		reset: bool = True,
		adr: str | None = None,
	) -> list[dict[str, float | None]]:
		"""Apply a constant current bias and measure voltage one or more times."""
		smu, temporary = self._get_session(adr=adr, timeout=30000)
		try:
			if reset:
				smu.write("*RST")
				smu.write("*CLS")
			smu.write(":ROUT:TERM REAR")
			smu.write(":SOUR:FUNC CURR")
			smu.write(':SENS:FUNC "VOLT"')
			smu.write(f":SENS:VOLT:PROT {float(voltage_compliance):.12g}")
			smu.write(f":SOUR:CURR:VLIM {float(voltage_compliance):.12g}")

			if current_range is not None:
				smu.write(":SOUR:CURR:RANG:AUTO OFF")
				smu.write(f":SOUR:CURR:RANG {float(current_range):.12g}")
			else:
				smu.write(":SOUR:CURR:RANG:AUTO ON")

			if voltage_range is not None:
				smu.write(":SENS:VOLT:RANG:AUTO OFF")
				smu.write(f":SENS:VOLT:RANG {float(voltage_range):.12g}")
			else:
				smu.write(":SENS:VOLT:RANG:AUTO ON")

			if nplc is not None:
				smu.write(f":SENS:VOLT:NPLC {float(nplc):.12g}")

			smu.write(":FORM:ELEM VOLT,CURR")
			smu.write(f":SOUR:CURR:LEV {float(current):.12g}")
			smu.write(":OUTP ON")

			if settle_s > 0:
				time.sleep(settle_s)

			t0 = time.perf_counter()
			records: list[dict[str, float | None]] = []
			for index in range(max(1, int(read_count))):
				v_meas, i_meas = self._read_measurement_pair(smu)
				records.append(
					_record(
						timestamp=time.perf_counter() - t0,
						i_cmd=current,
						v_meas=v_meas,
						i_meas=i_meas,
						cycle_number=float(index),
					)
				)

			smu.write(":OUTP OFF")
			return records
		finally:
			self._close_temp(smu, temporary)

	def source_voltage_measure_current(
		self,
		voltages: Iterable[float],
		current_compliance: float,
		delay_s: float = 0.0,
		current_range: float | None = None,
		voltage_range: float | None = None,
		reset: bool = True,
		adr: str | None = None,
		nplc: float | None = None,
		use_auto_current_range: bool = False,
		sweep_range_mode: str = "BEST",
	) -> list[dict[str, float | None]]:
		"""Run a native 2450 voltage-list sweep and read the trace buffer."""
		voltage_values = [float(value) for value in voltages]
		if not voltage_values:
			return []

		delay_value = max(0.0, float(delay_s))
		timeout_ms = max(120000, int((delay_value * len(voltage_values) + 30.0) * 1000))
		buffer_name = "defbuffer1"
		range_mode = str(sweep_range_mode).strip().upper()
		if range_mode not in {"AUTO", "BEST", "FIXED", "MANUAL"}:
			raise ValueError("sweep_range_mode must be AUTO, BEST, FIXED, or MANUAL.")

		smu, temporary = self._get_session(adr=adr, timeout=timeout_ms)
		try:
			if reset:
				smu.write("*RST")
				smu.write("*CLS")
			else:
				smu.write("*CLS")
			smu.write(":ROUT:TERM REAR")
			smu.write(":SOUR:FUNC VOLT")
			smu.write(":SOUR:VOLT:READ:BACK ON")
			smu.write(':SENS:FUNC "CURR"')
			smu.write(f":SENS:CURR:PROT {float(current_compliance):.12g}")
			smu.write(f":SOUR:VOLT:ILIM {float(current_compliance):.12g}")

			if voltage_range is not None:
				smu.write(":SOUR:VOLT:RANG:AUTO OFF")
				smu.write(f":SOUR:VOLT:RANG {float(voltage_range):.12g}")
			else:
				smu.write(":SOUR:VOLT:RANG:AUTO ON")

			if use_auto_current_range or range_mode == "AUTO":
				smu.write(":SENS:CURR:RANG:AUTO ON")
			elif current_range is not None:
				smu.write(":SENS:CURR:RANG:AUTO OFF")
				smu.write(f":SENS:CURR:RANG {float(current_range):.12g}")
			else:
				fixed_range = abs(float(current_compliance))
				if fixed_range <= 0:
					raise ValueError("current_compliance must be non-zero for fixed range.")
				smu.write(":SENS:CURR:RANG:AUTO OFF")
				smu.write(f":SENS:CURR:RANG {fixed_range:.12g}")

			if nplc is not None:
				smu.write(f":SENS:CURR:NPLC {float(nplc):.12g}")

			smu.write(f':TRAC:CLE "{buffer_name}"')
			smu.write(f':TRAC:POIN {len(voltage_values)}, "{buffer_name}"')
			smu.write(":FORM:DATA ASC")
			self._write_voltage_source_list(smu, voltage_values)
			smu.write(
				f':SOUR:SWE:VOLT:LIST 1, {delay_value:.12g}, 1, OFF, "{buffer_name}"'
			)
			smu.write(":OUTP ON")
			smu.write(":INIT")
			smu.query("*OPC?").strip()
			smu.write(":OUTP OFF")
			return self._read_trace_records(
				smu=smu,
				voltages=voltage_values,
				delay_s=delay_value,
				buffer_name=buffer_name,
			)
		finally:
			try:
				smu.write(":OUTP OFF")
			except Exception:
				pass
			self._close_temp(smu, temporary)

	def source_current_measure_voltage(
		self,
		currents: Iterable[float],
		voltage_compliance: float,
		delay_s: float = 0.0,
		current_range: float | None = None,
		voltage_range: float | None = None,
		reset: bool = True,
		adr: str | None = None,
		nplc: float | None = None,
	) -> list[dict[str, float | None]]:
		"""Step through a current list and measure voltage at each point."""
		smu, temporary = self._get_session(adr=adr, timeout=120000)
		try:
			if reset:
				smu.write("*RST")
				smu.write("*CLS")
			smu.write(":ROUT:TERM REAR")
			smu.write(":SOUR:FUNC CURR")
			smu.write(':SENS:FUNC "VOLT"')
			smu.write(f":SENS:VOLT:PROT {float(voltage_compliance):.12g}")
			smu.write(f":SOUR:CURR:VLIM {float(voltage_compliance):.12g}")

			if current_range is not None:
				smu.write(":SOUR:CURR:RANG:AUTO OFF")
				smu.write(f":SOUR:CURR:RANG {float(current_range):.12g}")
			else:
				smu.write(":SOUR:CURR:RANG:AUTO ON")

			if voltage_range is not None:
				smu.write(":SENS:VOLT:RANG:AUTO OFF")
				smu.write(f":SENS:VOLT:RANG {float(voltage_range):.12g}")
			else:
				smu.write(":SENS:VOLT:RANG:AUTO ON")

			if nplc is not None:
				smu.write(f":SENS:VOLT:NPLC {float(nplc):.12g}")

			smu.write(":FORM:ELEM VOLT,CURR")
			smu.write(":OUTP ON")

			t0 = time.perf_counter()
			records: list[dict[str, float | None]] = []
			for index, current in enumerate(float(i) for i in currents):
				smu.write(f":SOUR:CURR:LEV {current:.12g}")
				if delay_s > 0:
					time.sleep(delay_s)
				v_meas, i_meas = self._read_measurement_pair(smu)
				records.append(
					_record(
						timestamp=time.perf_counter() - t0,
						i_cmd=current,
						v_meas=v_meas,
						i_meas=i_meas,
						cycle_number=float(index),
					)
				)

			smu.write(":OUTP OFF")
			return records
		finally:
			self._close_temp(smu, temporary)

	def run_voltage_pulse_train(
		self,
		voltages: Sequence[float],
		current_compliance: float,
		pulse_width_s: float,
		acquire_delay_s: float | None = None,
		current_range: float | None = None,
		reset: bool = True,
		adr: str | None = None,
		current_autorange: bool = False,
	) -> list[dict[str, float | None]]:
		"""Approximate a pulse train using the 2450 native list-sweep model.

		The 2450 does not expose a dedicated fast pulse mode like the Keysight
		B2900 series. This method therefore constructs a source list and executes
		it with the internal list-sweep engine and reading buffer.
		"""
		_ = acquire_delay_s
		return self.source_voltage_measure_current(
			voltages=voltages,
			current_compliance=current_compliance,
			delay_s=pulse_width_s,
			current_range=current_range,
			reset=reset,
			adr=adr,
			use_auto_current_range=current_autorange,
		)


class KeithleySMU(Keithley2450):
	"""Backward-compatible alias used by legacy GUI code."""
