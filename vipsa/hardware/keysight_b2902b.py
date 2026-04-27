"""Pure PyVISA + native SCPI wrapper for the Keysight B2900/B2902B."""

# TODO: Resolve remaining legacy compatibility gaps: keep the old SMU/SCPI
# method surface stable while verifying command sequences on a real B2900/B2902B.

from __future__ import annotations

from typing import Any, Iterable, Sequence
import time
import warnings

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
			return "NO_PYVISA,KEYSIGHTB2902B,0,0"
		if command == "*OPC?":
			return "1"
		if command.startswith(":SYST:ERR?"):
			return "0,No error"
		if command.startswith(":FETC:ARR:SOUR"):
			return "0"
		if command.startswith(":FETC:ARR:VOLT"):
			return "0"
		if command.startswith(":FETC:ARR:CURR"):
			return "0"
		if command.startswith(":FETC:ARR:TIME"):
			return "0"
		if command.startswith(":READ?"):
			return "0,0"
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


class KeysightB2902B:
	"""Native SCPI driver for the Keysight B2900/B2902B family."""

	instrument_family = "keysight_b2902b"
	supports_native_pulse = True
	supports_dual_channel = True

	def __init__(
		self,
		device_no: int = 0,
		address: str | None = None,
		switch: Any | None = None,
		switch_channel: Any | None = None,
		connect_switch: bool = False,
		channel: int = 1,
	) -> None:
		"""Initialize the driver and open the main VISA session."""
		self.rm = _resource_manager()
		address_list = list(self.rm.list_resources())
		self.address = address if address is not None else address_list[device_no]
		self.switch = switch
		self.switch_channel = switch_channel
		self.switch_profile = "keysight"
		self.channel = int(channel)
		self.smu = self._open_resource(self.address, timeout=10000)

		if connect_switch and self.switch is not None:
			self.connect_switch_path()

		self.initialize()

	@property
	def _channel_suffix(self) -> str:
		"""Return the numeric channel suffix used by the active channel."""
		return str(self.channel)

	@property
	def _channel_list(self) -> str:
		"""Return the SCPI channel-list token for the active channel."""
		return f"(@{self.channel})"

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

	def _fetch_ascii_array(
		self,
		field: str,
		smu=None,
	) -> list[float]:
		"""Fetch one Keysight array field such as source, current, or time."""
		dev = self.smu if smu is None else smu
		name = str(field).strip().upper()
		if name in {"SOUR", "SOURCE"}:
			command = f":FETC:ARR:SOUR? {self._channel_list}"
		elif name in {"CURR", "CURRENT"}:
			command = f":FETC:ARR:CURR? {self._channel_list}"
		elif name in {"VOLT", "VOLTAGE"}:
			command = f":FETC:ARR:VOLT? {self._channel_list}"
		elif name in {"TIME", "TIM"}:
			command = f":FETC:ARR:TIME? {self._channel_list}"
		else:
			raise ValueError(f"Unsupported fetch field '{field}'.")

		raw = dev.query(command).strip()
		if not raw:
			return []
		return [float(part.strip()) for part in raw.split(",") if part.strip()]

	def _configure_simple_read(
		self,
		source_function: str,
		sense_function: str,
		smu,
	) -> None:
		"""Apply a simple single-channel source/measure pairing."""
		src = self._normalize_function(source_function)
		sns = self._normalize_function(sense_function)
		suffix = self._channel_suffix
		smu.write(f":SOUR{suffix}:FUNC:MODE {src}")
		smu.write(f':SENS{suffix}:FUNC "{sns}"')

	def _warn_deprecated_protocol(self, method_name: str) -> None:
		"""Warn once when a driver-level compound helper delegates upward."""
		attr_name = f"_warned_protocol_{method_name}"
		if getattr(self, attr_name, False):
			return
		warnings.warn(
			f"{self.__class__.__name__}.{method_name} is deprecated at the driver layer; "
			"call the SourceMeasureUnit orchestration method instead. "
			"Delegating for compatibility.",
			DeprecationWarning,
			stacklevel=3,
		)
		setattr(self, attr_name, True)

	def _build_orchestration_adapter(self):
		"""Create a temporary orchestration wrapper around this live driver."""
		existing = getattr(self, "orchestrator", None)
		if existing is not None:
			return existing
		try:
			from .Source_Measure_Unit import SourceMeasureUnit
		except ImportError:
			from Source_Measure_Unit import SourceMeasureUnit

		handler = SourceMeasureUnit(
			driver=self,
			tiny_iv_path=getattr(self, "tiny_IV", None),
		)
		if getattr(self, "resistance_df", None) is not None:
			handler.resistance_df = self.resistance_df
		return handler

	def _delegate_protocol_method(self, method_name: str, *args: Any, **kwargs: Any):
		"""Delegate a compound measurement helper to SourceMeasureUnit."""
		self._warn_deprecated_protocol(method_name)
		handler = self._build_orchestration_adapter()
		return getattr(handler, method_name)(*args, **kwargs)

	def initialize(self) -> "KeysightB2902B":
		"""Apply a conservative default state for lab automation."""
		self.write("*RST")
		self.write("*CLS")
		self.write(f":SOUR{self._channel_suffix}:FUNC:MODE VOLT")
		self.write(f':SENS{self._channel_suffix}:FUNC "CURR"')
		self.write(f":OUTP{self._channel_suffix} OFF")
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
	) -> "KeysightB2902B":
		"""Open or reopen the persistent Keysight VISA session."""
		if address is not None:
			self.address = address
		elif self.smu is None:
			addresses = list(self.rm.list_resources())
			self.address = addresses[device_no]

		if self.smu is None:
			self.smu = self._open_resource(self.address, timeout=10000)
		return self

	def disconnect(self) -> None:
		"""Close the active Keysight session using the legacy lifecycle name."""
		self.close_session()

	def clear(self) -> "KeysightB2902B":
		"""Clear the Keysight status and error queues."""
		self.write("*CLS")
		return self

	def reset(self) -> "KeysightB2902B":
		"""Reset the Keysight and reapply the ViPSA default setup."""
		self.write("*RST")
		self.clear()
		self.initialize()
		return self

	def set_source_function(self, function: str) -> None:
		"""Select voltage or current source mode."""
		func = self._normalize_function(function)
		self.write(f":SOUR{self._channel_suffix}:FUNC:MODE {func}")

	def set_sense_function(self, function: str | Sequence[str]) -> None:
		"""Enable one or more sense functions.

		The B2900 family supports a primary sense function for the channel.
		When multiple functions are supplied, the first entry is used.
		"""
		if isinstance(function, (list, tuple)):
			selected = function[0]
		else:
			selected = function
		func = self._normalize_function(str(selected))
		self.write(f':SENS{self._channel_suffix}:FUNC "{func}"')

	def set_source_range(
		self,
		function: str,
		value: float | None = None,
		auto: bool = True,
	) -> None:
		"""Configure the source range or source autorange."""
		func = self._normalize_function(function)
		if auto:
			self.write(f":SOUR{self._channel_suffix}:{func}:RANG:AUTO ON")
			return
		if value is None:
			raise ValueError("A source range value is required when auto is False.")
		self.write(f":SOUR{self._channel_suffix}:{func}:RANG:AUTO OFF")
		self.write(f":SOUR{self._channel_suffix}:{func}:RANG {float(value):.12g}")

	def set_sense_range(
		self,
		function: str,
		value: float | None = None,
		auto: bool = True,
	) -> None:
		"""Configure the measurement range or measurement autorange."""
		func = self._normalize_function(function)
		if auto:
			self.write(f":SENS{self._channel_suffix}:{func}:RANG:AUTO ON")
			return
		if value is None:
			raise ValueError("A sense range value is required when auto is False.")
		self.write(f":SENS{self._channel_suffix}:{func}:RANG:AUTO OFF")
		self.write(f":SENS{self._channel_suffix}:{func}:RANG {float(value):.12g}")

	def set_protection(self, function: str, value: float) -> None:
		"""Configure current or voltage protection / compliance."""
		func = self._normalize_function(function)
		numeric = float(value)
		if func == "VOLT":
			self.write(f":SENS{self._channel_suffix}:CURR:PROT {numeric:.12g}")
			self.write(f":SOUR{self._channel_suffix}:VOLT:ILIM {numeric:.12g}")
			return
		self.write(f":SENS{self._channel_suffix}:VOLT:PROT {numeric:.12g}")
		self.write(f":SOUR{self._channel_suffix}:CURR:VLIM {numeric:.12g}")

	def set_source_level(self, function: str, value: float) -> None:
		"""Set the immediate source level for voltage or current."""
		func = self._normalize_function(function)
		self.write(f":SOUR{self._channel_suffix}:{func} {float(value):.12g}")

	def set_output(self, enabled: bool) -> None:
		"""Turn the output on or off for the active channel."""
		self.write(
			f":OUTP{self._channel_suffix} {'ON' if enabled else 'OFF'}"
		)

	def configure_measurement_format(self, functions: str | Sequence[str]) -> None:
		"""Configure the array and scalar response field order."""
		if isinstance(functions, str):
			payload = functions
		else:
			payload = ",".join(str(item).strip().upper() for item in functions)
		self.write(f":FORM:ELEM:SENS {payload}")

	def query_reading(self) -> str:
		"""Query one reading from the active source-measure setup."""
		return self.query(f":READ? {self._channel_list}")

	def connect_switch_path(self) -> None:
		"""Connect the assigned switch path if a switch wrapper is available.

		Best effort: force the SMU output off before changing the active route.
		"""
		if self.switch is None:
			return
		self._prepare_for_route_change()

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
		"""Open the assigned switch path if a switch wrapper is available.

		Best effort: force the SMU output off before changing the active route.
		"""
		if self.switch is None:
			return
		self._prepare_for_route_change()

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

	def reset_device(self, adr: str | None = None) -> None:
		"""Reset and reinitialize the instrument."""
		smu, temporary = self._get_session(adr=adr, timeout=10000)
		try:
			smu.write("*RST")
			smu.write("*CLS")
			smu.write(f":SOUR{self._channel_suffix}:FUNC:MODE VOLT")
			smu.write(f':SENS{self._channel_suffix}:FUNC "CURR"')
			smu.write(f":OUTP{self._channel_suffix} OFF")
		finally:
			self._close_temp(smu, temporary)

	def send_command(self, command: str, adr: str | None = None) -> None:
		"""Backward-compatible helper that sends one raw command."""
		self.write(command, adr=adr)

	def sync(self, SMU: Any | None = None) -> Any:
		"""Wait for operation completion using ``*OPC?``."""
		dev = self.smu if SMU is None else SMU
		return dev.query("*OPC?").strip()

	def get_error(self, SMU: Any | None = None) -> list[str]:
		"""Return all currently queued instrument errors."""
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

	def stop_output(self, adr: str | None = None) -> None:
		"""Turn the channel output off."""
		smu, temporary = self._get_session(adr=adr, timeout=5000)
		try:
			smu.write(f":OUTP{self._channel_suffix} OFF")
		finally:
			self._close_temp(smu, temporary)

	def _prepare_for_route_change(self) -> None:
		"""Best-effort safety step before switch-routing changes."""
		try:
			self.abort_measurement()
			return
		except Exception:
			pass
		try:
			self.stop_output()
		except Exception:
			pass

	def abort_measurement(self, adr: str | None = None) -> None:
		"""Abort active operations and leave the output off."""
		smu, temporary = self._get_session(adr=adr, timeout=5000)
		try:
			try:
				smu.write(":ABOR")
			except Exception:
				pass
			try:
				smu.write(f":OUTP{self._channel_suffix} OFF")
			except Exception:
				pass
		finally:
			self._close_temp(smu, temporary)

	def prepare_contact_probe(
		self,
		voltage: float,
		compliance: float,
		adr: str | None = None,
	) -> None:
		"""Prepare a steady voltage bias for contact-current probing."""
		smu, temporary = self._get_session(adr=adr, timeout=10000)
		try:
			self.connect_switch_path()
			smu.write("*RST")
			smu.write("*CLS")
			smu.write(f":SOUR{self._channel_suffix}:FUNC:MODE VOLT")
			smu.write(f':SENS{self._channel_suffix}:FUNC "CURR"')
			smu.write(f":SENS{self._channel_suffix}:CURR:PROT {float(compliance):.12g}")
			smu.write(f":SOUR{self._channel_suffix}:VOLT {float(voltage):.12g}")
			smu.write(f":OUTP{self._channel_suffix} ON")
		finally:
			self._close_temp(smu, temporary)

	def get_contact_current(
		self,
		voltage: float,
		compliance: float = 10e-6,
		adr: str | None = None,
	) -> float:
		"""Apply one voltage bias and return the absolute measured current."""
		records = self.hold_voltage_measure_current(
			voltage=voltage,
			current_compliance=compliance,
			settle_s=0.02,
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
		adr: str | None = None,
		settle: float = 0.02,
	) -> float:
		"""Read absolute current quickly assuming the instrument is configured."""
		smu, temporary = self._get_session(adr=adr, timeout=5000)
		try:
			smu.write(f":SOUR{self._channel_suffix}:VOLT {float(voltage):.12g}")
			smu.write(f":OUTP{self._channel_suffix} ON")
			if settle > 0:
				time.sleep(settle)
			response = smu.query(f":READ? {self._channel_list}").strip()
			parts = [part.strip() for part in response.split(",") if part.strip()]
			if len(parts) >= 2:
				return abs(float(parts[1]))
			if len(parts) == 1:
				return abs(float(parts[0]))
			return 0.0
		finally:
			self._close_temp(smu, temporary)

	def read_current_at_voltage(
		self,
		voltage: float,
		current_compliance: float = 10e-6,
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
		current_compliance: float = 10e-6,
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
		current_compliance: float | None = None,
		set_width: float = 0.01,
		bare_list: Sequence[float] | None = None,
		set_acquire_delay: float | None = None,
		adr: str | None = None,
		compliance: float | None = None,
		pulse_width: float | None = None,
		current_autorange: bool = False,
	) -> list[dict[str, float | None]]:
		"""Deprecated compound wrapper delegated to SourceMeasureUnit."""
		return self._delegate_protocol_method(
			"pulsed_measurement",
			csv_path=csv_path,
			current_compliance=current_compliance,
			set_width=set_width,
			bare_list=bare_list,
			set_acquire_delay=set_acquire_delay,
			adr=adr,
			compliance=compliance,
			pulse_width=pulse_width,
			current_autorange=current_autorange,
		)

	def response_dealer(self, raw_response: str) -> dict[str, list[float]]:
		"""Deprecated parsing helper delegated to SourceMeasureUnit."""
		return self._delegate_protocol_method("response_dealer", raw_response)

	def split_pulse_for_2_chan(
		self,
		vlist: Sequence[float],
	) -> tuple[list[float], list[float]]:
		"""Deprecated pulse-splitting helper delegated to SourceMeasureUnit."""
		return self._delegate_protocol_method("split_pulse_for_2_chan", vlist)

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
			suffix = self._channel_suffix
			smu.write(f":SOUR{suffix}:FUNC:MODE VOLT")
			smu.write(f':SENS{suffix}:FUNC "CURR"')
			smu.write(f":SENS{suffix}:CURR:PROT {float(current_compliance):.12g}")

			if voltage_range is not None:
				smu.write(f":SOUR{suffix}:VOLT:RANG:AUTO OFF")
				smu.write(f":SOUR{suffix}:VOLT:RANG {float(voltage_range):.12g}")
			else:
				smu.write(f":SOUR{suffix}:VOLT:RANG:AUTO ON")

			if current_range is not None:
				smu.write(f":SENS{suffix}:CURR:RANG:AUTO OFF")
				smu.write(f":SENS{suffix}:CURR:RANG {float(current_range):.12g}")
			else:
				smu.write(f":SENS{suffix}:CURR:RANG:AUTO ON")

			if nplc is not None:
				smu.write(f":SENS{suffix}:CURR:NPLC {float(nplc):.12g}")

			smu.write(":FORM:ELEM:SENS VOLT,CURR")
			smu.write(f":SOUR{suffix}:VOLT {float(voltage):.12g}")
			smu.write(f":OUTP{suffix} ON")

			if settle_s > 0:
				time.sleep(settle_s)

			t0 = time.perf_counter()
			records: list[dict[str, float | None]] = []
			for index in range(max(1, int(read_count))):
				response = smu.query(f":READ? {self._channel_list}").strip()
				parts = [part.strip() for part in response.split(",") if part.strip()]
				v_meas = float(parts[0]) if len(parts) >= 1 else None
				i_meas = float(parts[1]) if len(parts) >= 2 else None
				records.append(
					_record(
						timestamp=time.perf_counter() - t0,
						v_cmd=voltage,
						v_meas=v_meas,
						i_meas=i_meas,
						cycle_number=float(index),
					)
				)

			smu.write(f":OUTP{suffix} OFF")
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
			suffix = self._channel_suffix
			smu.write(f":SOUR{suffix}:FUNC:MODE CURR")
			smu.write(f':SENS{suffix}:FUNC "VOLT"')
			smu.write(f":SENS{suffix}:VOLT:PROT {float(voltage_compliance):.12g}")

			if current_range is not None:
				smu.write(f":SOUR{suffix}:CURR:RANG:AUTO OFF")
				smu.write(f":SOUR{suffix}:CURR:RANG {float(current_range):.12g}")
			else:
				smu.write(f":SOUR{suffix}:CURR:RANG:AUTO ON")

			if voltage_range is not None:
				smu.write(f":SENS{suffix}:VOLT:RANG:AUTO OFF")
				smu.write(f":SENS{suffix}:VOLT:RANG {float(voltage_range):.12g}")
			else:
				smu.write(f":SENS{suffix}:VOLT:RANG:AUTO ON")

			if nplc is not None:
				smu.write(f":SENS{suffix}:VOLT:NPLC {float(nplc):.12g}")

			smu.write(":FORM:ELEM:SENS VOLT,CURR")
			smu.write(f":SOUR{suffix}:CURR {float(current):.12g}")
			smu.write(f":OUTP{suffix} ON")

			if settle_s > 0:
				time.sleep(settle_s)

			t0 = time.perf_counter()
			records: list[dict[str, float | None]] = []
			for index in range(max(1, int(read_count))):
				response = smu.query(f":READ? {self._channel_list}").strip()
				parts = [part.strip() for part in response.split(",") if part.strip()]
				v_meas = float(parts[0]) if len(parts) >= 1 else None
				i_meas = float(parts[1]) if len(parts) >= 2 else None
				records.append(
					_record(
						timestamp=time.perf_counter() - t0,
						i_cmd=current,
						v_meas=v_meas,
						i_meas=i_meas,
						cycle_number=float(index),
					)
				)

			smu.write(f":OUTP{suffix} OFF")
			return records
		finally:
			self._close_temp(smu, temporary)

	def source_voltage_measure_current(
		self,
		voltages: Sequence[float],
		current_compliance: float,
		delay_s: float = 0.0,
		current_range: float | None = None,
		voltage_range: float | None = None,
		reset: bool = True,
		use_auto_current_range: bool = False,
		sweep_range_mode: str = "BEST",
		adr: str | None = None,
	) -> list[dict[str, float | None]]:
		"""Run a voltage-list IV primitive using Keysight list mode."""
		voltage_values = [float(value) for value in voltages]
		if not voltage_values:
			return []

		delay_value = max(0.0, float(delay_s))
		timeout_ms = max(120000, int((delay_value * len(voltage_values) + 30.0) * 1000))
		range_mode = str(sweep_range_mode).strip().upper()
		if range_mode not in {"AUTO", "BEST", "FIXED", "MANUAL"}:
			raise ValueError("sweep_range_mode must be AUTO, BEST, FIXED, or MANUAL.")
		if range_mode == "MANUAL":
			range_mode = "FIXED"

		smu, temporary = self._get_session(adr=adr, timeout=timeout_ms)
		try:
			suffix = self._channel_suffix
			voltage_csv = ",".join(f"{voltage:.12g}" for voltage in voltage_values)

			if reset:
				smu.write("*RST")
				smu.write("*CLS")
			else:
				smu.write("*CLS")

			smu.write(":ROUT:TERM REAR")
			smu.write(":FORM:ELEM:SENS VOLT,CURR,TIME,SOUR")
			smu.write(f":SOUR{suffix}:FUNC:MODE VOLT")
			smu.write(f":SOUR{suffix}:VOLT:MODE LIST")
			smu.write(f':SENS{suffix}:FUNC "CURR","VOLT"')
			smu.write(f":SENS{suffix}:VOLT:RANG:AUTO ON")
			smu.write(f":SENS{suffix}:CURR:PROT {float(current_compliance):.12g}")

			if voltage_range is not None:
				smu.write(f":SOUR{suffix}:VOLT:RANG:AUTO OFF")
				smu.write(f":SOUR{suffix}:VOLT:RANG {float(voltage_range):.12g}")
			else:
				smu.write(f":SOUR{suffix}:VOLT:RANG:AUTO ON")

			if use_auto_current_range:
				smu.write(f":SENS{suffix}:CURR:RANG:AUTO ON")
			elif current_range is not None:
				smu.write(f":SENS{suffix}:CURR:RANG:AUTO OFF")
				smu.write(f":SENS{suffix}:CURR:RANG {float(current_range):.12g}")
			else:
				fixed_range = abs(float(current_compliance))
				if fixed_range <= 0:
					raise ValueError("current_compliance must be non-zero for fixed range.")
				smu.write(f":SENS{suffix}:CURR:RANG:AUTO OFF")
				smu.write(f":SENS{suffix}:CURR:RANG {fixed_range:.12g}")

			smu.write(f":SOUR{suffix}:LIST:VOLT {voltage_csv}")
			smu.write(f":SOUR{suffix}:SWE:POIN {len(voltage_values)}")
			smu.write(f":SOUR{suffix}:SWE:RANG {range_mode}")
			smu.write(f":TRIG{suffix}:TRAN:DEL 0")
			smu.write(f":TRIG{suffix}:SOUR TIM")
			smu.write(f":TRIG{suffix}:TIM {delay_value:.12g}")
			smu.write(f":TRIG{suffix}:COUN {len(voltage_values)}")
			smu.write(f":TRAC{suffix}:FEED:CONT NEV")
			smu.write(f":TRAC{suffix}:CLE")
			smu.write(f":TRAC{suffix}:POIN {len(voltage_values)}")
			smu.write(f":TRAC{suffix}:FEED SENS")
			smu.write(f":TRAC{suffix}:FEED:CONT NEXT")

			smu.write(f":OUTP{suffix} ON")
			smu.write(f":INIT {self._channel_list}")
			smu.query("*OPC?").strip()
			smu.write(f":OUTP{suffix} OFF")

			source_values = self._fetch_ascii_array("SOUR", smu=smu)
			voltage_readings = self._fetch_ascii_array("VOLT", smu=smu)
			current_values = self._fetch_ascii_array("CURR", smu=smu)
			time_values = self._fetch_ascii_array("TIME", smu=smu)

			records: list[dict[str, float | None]] = []
			for index, voltage in enumerate(voltage_values):
				v_cmd = source_values[index] if index < len(source_values) else voltage
				v_meas = voltage_readings[index] if index < len(voltage_readings) else None
				i_meas = current_values[index] if index < len(current_values) else None
				timestamp = time_values[index] if index < len(time_values) else float(index) * delay_value
				records.append(
					_record(
						timestamp=timestamp,
						v_cmd=v_cmd,
						v_meas=v_meas,
						i_meas=i_meas,
						cycle_number=float(index),
					)
				)

			return records
		finally:
			try:
				smu.write(f":OUTP{self._channel_suffix} OFF")
			except Exception:
				pass
			self._close_temp(smu, temporary)

	def source_current_measure_voltage(
		self,
		currents: Sequence[float],
		voltage_compliance: float,
		delay_s: float = 0.0,
		current_range: float | None = None,
		voltage_range: float | None = None,
		reset: bool = True,
		adr: str | None = None,
	) -> list[dict[str, float | None]]:
		"""Run a current-list IV primitive using Keysight list mode."""
		smu, temporary = self._get_session(adr=adr, timeout=120000)
		try:
			if not currents:
				return []

			suffix = self._channel_suffix
			current_csv = ",".join(f"{float(i):.12g}" for i in currents)
			delay_value = float(delay_s)

			if reset:
				smu.write("*RST")
				smu.write("*CLS")

			smu.write(":FORM:ELEM:SENS VOLT,CURR,TIME")
			smu.write(f":SOUR{suffix}:FUNC:MODE CURR")
			smu.write(f":SOUR{suffix}:CURR:MODE LIST")
			smu.write(f":SOUR{suffix}:LIST:CURR {current_csv}")
			smu.write(f':SENS{suffix}:FUNC "VOLT"')
			smu.write(f":SENS{suffix}:VOLT:PROT {float(voltage_compliance):.12g}")

			if current_range is not None:
				smu.write(f":SOUR{suffix}:CURR:RANG:AUTO OFF")
				smu.write(f":SOUR{suffix}:CURR:RANG {float(current_range):.12g}")
			else:
				smu.write(f":SOUR{suffix}:CURR:RANG:AUTO ON")

			if voltage_range is not None:
				smu.write(f":SENS{suffix}:VOLT:RANG:AUTO OFF")
				smu.write(f":SENS{suffix}:VOLT:RANG {float(voltage_range):.12g}")
			else:
				smu.write(f":SENS{suffix}:VOLT:RANG:AUTO ON")

			smu.write(f":TRIG:TRAN:DEL 0")
			smu.write(":TRIG:SOUR TIM")
			smu.write(f":TRIG:TIM {delay_value:.12g}")
			smu.write(f":TRIG:COUN {len(currents)}")
			smu.write(f":SOUR{suffix}:SWE:POIN {len(currents)}")

			smu.write(f":OUTP{suffix} ON")
			smu.write(f":INIT {self._channel_list}")
			smu.query("*OPC?").strip()
			smu.write(f":OUTP{suffix} OFF")

			source_values = self._fetch_ascii_array("CURR", smu=smu)
			voltage_values = self._fetch_ascii_array("VOLT", smu=smu)
			time_values = self._fetch_ascii_array("TIME", smu=smu)

			records: list[dict[str, float | None]] = []
			for index, current in enumerate(currents):
				i_cmd = source_values[index] if index < len(source_values) else float(current)
				v_meas = voltage_values[index] if index < len(voltage_values) else None
				timestamp = time_values[index] if index < len(time_values) else float(index) * delay_value
				records.append(
					_record(
						timestamp=timestamp,
						i_cmd=i_cmd,
						v_meas=v_meas,
						cycle_number=float(index),
					)
				)

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
		"""Run a native Keysight pulse/list train on the active channel."""
		smu, temporary = self._get_session(adr=adr, timeout=120000)
		try:
			if not voltages:
				return []

			suffix = self._channel_suffix
			pulse_width = float(pulse_width_s)
			acq_delay = pulse_width / 2 if acquire_delay_s is None else float(acquire_delay_s)
			acq_delay = max(0.0, min(pulse_width, acq_delay))
			voltage_csv = ",".join(f"{float(v):.12g}" for v in voltages)

			if reset:
				smu.write("*RST")
				smu.write("*CLS")
			else:
				smu.write("*CLS")

			smu.write(":ROUT:TERM REAR")
			smu.write(":FORM:ELEM:SENS VOLT,CURR,TIME,SOUR")
			smu.write(f":SOUR{suffix}:FUNC:MODE VOLT")
			smu.write(f":SOUR{suffix}:FUNC:SHAP PULS")
			smu.write(f":SOUR{suffix}:VOLT:MODE LIST")
			smu.write(f":SOUR{suffix}:LIST:VOLT {voltage_csv}")
			smu.write(f':SENS{suffix}:FUNC "CURR","VOLT"')
			smu.write(f":SENS{suffix}:VOLT:RANG:AUTO ON")
			smu.write(f":SENS{suffix}:CURR:PROT {float(current_compliance):.12g}")

			if current_autorange:
				smu.write(f":SENS{suffix}:CURR:RANG:AUTO ON")
			elif current_range is not None:
				smu.write(f":SENS{suffix}:CURR:RANG:AUTO OFF")
				smu.write(f":SENS{suffix}:CURR:RANG {float(current_range):.12g}")
			else:
				fixed_range = abs(float(current_compliance))
				if fixed_range <= 0:
					raise ValueError("current_compliance must be non-zero for fixed range.")
				smu.write(f":SENS{suffix}:CURR:RANG:AUTO OFF")
				smu.write(f":SENS{suffix}:CURR:RANG {fixed_range:.12g}")

			smu.write(f":SOUR{suffix}:PULS:WIDT {pulse_width:.12g}")
			smu.write(f":SOUR{suffix}:PULS:DEL 0")
			smu.write(f":TRIG{suffix}:TRAN:DEL 0")
			smu.write(f":TRIG{suffix}:ACQ:DEL {acq_delay:.12g}")
			smu.write(f":TRIG{suffix}:TIM {pulse_width:.12g}")
			smu.write(f":TRIG{suffix}:COUN {len(voltages)}")
			smu.write(f":TRAC{suffix}:FEED:CONT NEV")
			smu.write(f":TRAC{suffix}:CLE")
			smu.write(f":TRAC{suffix}:POIN {len(voltages)}")
			smu.write(f":TRAC{suffix}:FEED SENS")
			smu.write(f":TRAC{suffix}:FEED:CONT NEXT")

			smu.write(f":OUTP{suffix} ON")
			smu.write(f":INIT {self._channel_list}")
			smu.query("*OPC?").strip()
			smu.write(f":OUTP{suffix} OFF")

			source_values = self._fetch_ascii_array("SOUR", smu=smu)
			voltage_readings = self._fetch_ascii_array("VOLT", smu=smu)
			current_values = self._fetch_ascii_array("CURR", smu=smu)
			time_values = self._fetch_ascii_array("TIME", smu=smu)

			records: list[dict[str, float | None]] = []
			for index, voltage in enumerate(voltages):
				v_cmd = source_values[index] if index < len(source_values) else float(voltage)
				v_meas = voltage_readings[index] if index < len(voltage_readings) else None
				i_meas = current_values[index] if index < len(current_values) else None
				timestamp = time_values[index] if index < len(time_values) else float(index) * pulse_width
				records.append(
					_record(
						timestamp=timestamp,
						v_cmd=v_cmd,
						v_meas=v_meas,
						i_meas=i_meas,
						cycle_number=float(index),
					)
				)

			return records
		finally:
			try:
				smu.write(f":OUTP{self._channel_suffix} OFF")
			except Exception:
				pass
			self._close_temp(smu, temporary)


class KeysightSMU(KeysightB2902B):
	"""Backward-compatible alias used by legacy GUI code."""

	def __init__(
		self,
		device_no: int = 0,
		address: str | None = None,
		switch: Any | None = None,
		switch_channel: Any | None = None,
		connect_switch: bool = False,
	) -> None:
		"""Initialize the backward-compatible single-channel wrapper."""
		super().__init__(
			device_no=device_no,
			address=address,
			switch=switch,
			switch_channel=switch_channel,
			connect_switch=connect_switch,
			channel=1,
		)
