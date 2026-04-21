"""Universal Source/Measure orchestration layer built on pure PyVISA drivers."""

# TODO: Resolve remaining legacy compatibility gaps: keep old GUI/workflow call
# signatures stable while tightening the canonical SMU orchestration API.

from __future__ import annotations

from typing import Any, Sequence
import time

import numpy as np

try:
	import pyvisa
except ImportError:
	try:
		from .SCPI import pyvisa
	except ImportError:
		from SCPI import pyvisa

try:
	import pandas as pd
except ModuleNotFoundError:
	class _MissingPandas:
		"""Small import-time placeholder when pandas is not installed."""

		def read_csv(self, *args: Any, **kwargs: Any):
			raise ModuleNotFoundError(
				"pandas is required for CSV-backed SMU operations."
			)

		class DataFrame:
			"""Placeholder DataFrame that raises when pandas-backed data is used."""

			def __init__(self, *args: Any, **kwargs: Any) -> None:
				raise ModuleNotFoundError(
					"pandas is required for DataFrame-backed SMU operations."
				)

			@classmethod
			def from_records(cls, *args: Any, **kwargs: Any):
				raise ModuleNotFoundError(
					"pandas is required for DataFrame-backed SMU operations."
				)

	pd = _MissingPandas()

try:
	from .keithley_2450 import Keithley2450
	from .keithley_707b import Keithley707B
	from .keysight_b2902b import KeysightB2902B
except ImportError:
	from keithley_2450 import Keithley2450
	from keithley_707b import Keithley707B
	from keysight_b2902b import KeysightB2902B


def _warn_once(obj: Any, attr_name: str, message: str) -> None:
	"""Print one warning once per object instance."""
	if getattr(obj, attr_name, False):
		return
	print(message)
	setattr(obj, attr_name, True)


def _measurement_record(
	timestamp: float,
	v_cmd: float | None,
	current: float | None,
	v_meas: float | None = None,
	cycle_number: float | None = None,
) -> dict[str, float | None]:
	"""Build one normalized compatibility record."""
	v_error = None
	if v_cmd is not None and v_meas is not None:
		v_error = float(v_meas) - float(v_cmd)

	return {
		"Time(T)": float(timestamp),
		"Voltage (V)": float(v_cmd) if v_cmd is not None else v_meas,
		"Current (A)": float(current) if current is not None else None,
		"V_cmd (V)": float(v_cmd) if v_cmd is not None else None,
		"V_meas (V)": float(v_meas) if v_meas is not None else None,
		"V_error (V)": v_error,
		"Cycle Number": cycle_number,
	}


def _records_to_frame(records: list[dict[str, float | None]]) -> pd.DataFrame:
	"""Convert compatibility records into a predictable DataFrame."""
	columns = [
		"Time(T)",
		"Voltage (V)",
		"Current (A)",
		"V_cmd (V)",
		"V_meas (V)",
		"V_error (V)",
		"Cycle Number",
	]
	if not records:
		return pd.DataFrame(columns=columns)

	frame = pd.DataFrame.from_records(records)
	for column in columns:
		if column not in frame.columns:
			frame[column] = np.nan
	return frame[columns]


def _records_to_legacy_array(records: list[dict[str, float | None]]) -> np.ndarray:
	"""Convert compatibility records into the legacy ``[t, v, i]`` array."""
	if not records:
		return np.empty((0, 3), dtype=float)

	rows = []
	for record in records:
		rows.append(
			[
				record.get("Time(T)", np.nan),
				record.get("Voltage (V)", np.nan),
				record.get("Current (A)", np.nan),
			]
		)
	return np.asarray(rows, dtype=float)


class SourceMeasureUnit:
	"""Vendor-agnostic orchestration layer around a driver instance.

	The driver object is expected to expose a small set of source/measure
	primitives implemented with raw PyVISA + SCPI.
	"""

	def __init__(
		self,
		driver: Any,
		tiny_iv_path: str | None = None,
	) -> None:
		"""Store the driver and optionally load the default tiny-IV profile."""
		self.driver = driver
		self.address = getattr(driver, "address", None)
		self.switch = getattr(driver, "switch", None)
		self.switch_channel = getattr(driver, "switch_channel", None)
		self.switch_profile = getattr(driver, "switch_profile", None)
		self.instrument_family = getattr(driver, "instrument_family", "unknown")
		self.supports_native_pulse = bool(
			getattr(driver, "supports_native_pulse", False)
		)

		self.tiny_IV = tiny_iv_path
		self.resistance_df = None
		if tiny_iv_path is not None:
			self.resistance_df = pd.read_csv(tiny_iv_path)

	def __getattr__(self, name: str) -> Any:
		"""Delegate unknown attributes to the underlying driver."""
		return getattr(self.driver, name)

	def _coerce_voltage_list(
		self,
		csv_path: str | None,
		bare_list: Sequence[float] | None,
	) -> tuple[list[float], list[float] | None]:
		"""Load a voltage list either from an explicit sequence or from CSV."""
		if bare_list is not None:
			return [float(value) for value in bare_list], None

		if csv_path is None:
			raise ValueError("Either csv_path or bare_list must be supplied.")

		frame = pd.read_csv(csv_path, dtype=float)
		times = frame.iloc[:, 0].astype(float).tolist()
		voltages = frame.iloc[:, 1].astype(float).tolist()
		return voltages, times

	def _resolve_delay_and_acq(
		self,
		times: list[float] | None,
		delay: float | None,
		acq_delay: float | None,
		default_delay: float = 0.01,
	) -> tuple[float, float]:
		"""Resolve the point spacing and acquisition delay."""
		if delay is None:
			if times is not None and len(times) >= 2:
				delay = float(times[1]) - float(times[0])
			else:
				delay = default_delay

		if acq_delay is None:
			acq_delay = float(delay) / 2

		return float(delay), float(acq_delay)

	def _driver_records_to_compatibility(
		self,
		records: list[dict[str, float | None]],
		cycle_number: float | None = None,
	) -> list[dict[str, float | None]]:
		"""Convert driver-native records into the old compatibility format."""
		compatibility_records: list[dict[str, float | None]] = []
		for index, record in enumerate(records):
			compatibility_records.append(
				_measurement_record(
					timestamp=float(record.get("Time(T)", float(index))),
					v_cmd=record.get("V_cmd (V)", record.get("Voltage (V)")),
					current=record.get("Current (A)", record.get("I_meas (A)")),
					v_meas=record.get("V_meas (V)"),
					cycle_number=cycle_number
					if cycle_number is not None
					else record.get("Cycle Number"),
				)
			)
		return compatibility_records

	def split_list(self, vlist: Sequence[float]) -> list[list[Any]]:
		"""Split a voltage list into positive and negative contiguous segments."""
		voltages = [float(value) for value in vlist]
		if not voltages:
			return []

		voltage_data: list[list[Any]] = []
		cycle_no = 0
		current_cycle: list[float] = []
		current_tag = "p" if voltages[0] >= 0 else "n"

		for index, voltage in enumerate(voltages):
			if index == 0:
				current_cycle.append(voltage)
				continue

			next_tag = "p" if voltage >= 0 else "n"
			if next_tag != current_tag:
				cycle_no += 1
				voltage_data.append([cycle_no, current_tag, current_cycle])
				current_tag = next_tag
				current_cycle = [voltage]
				continue

			current_cycle.append(voltage)

		if current_cycle:
			cycle_no += 1
			voltage_data.append([cycle_no, current_tag, current_cycle])

		return voltage_data

	def split_list_by_4(self, vlist: Sequence[float]) -> list[list[Any]]:
		"""Split a voltage list into positive/negative forward/backward segments."""
		voltages = [float(value) for value in vlist]
		if not voltages:
			return []

		segments: list[list[Any]] = []
		cycle_number = 0
		current_segment: list[float] = [voltages[0]]

		def classify(previous: float, current: float) -> str:
			if current >= 0:
				return "pf" if current >= previous else "pb"
			return "nf" if current <= previous else "nb"

		current_tag = "pf" if voltages[0] >= 0 else "nf"

		for previous, current in zip(voltages[:-1], voltages[1:]):
			tag = classify(previous, current)
			if tag != current_tag and current_segment:
				cycle_number += 1
				segments.append([cycle_number, current_tag, current_segment])
				current_segment = [current]
				current_tag = tag
			else:
				current_segment.append(current)

		if current_segment:
			cycle_number += 1
			segments.append([cycle_number, current_tag, current_segment])

		return segments

	def split_pulse_for_2_chan(
		self,
		vlist: Sequence[float],
	) -> tuple[list[float], list[float]]:
		"""Split signed voltages into positive-channel and negative-channel lists."""
		vlist_p: list[float] = []
		vlist_n: list[float] = []

		for voltage in [float(value) for value in vlist]:
			if voltage >= 0:
				vlist_p.append(voltage)
				vlist_n.append(0.0)
			else:
				vlist_p.append(0.0)
				vlist_n.append(abs(voltage))

		return vlist_p, vlist_n

	def response_dealer(self, raw_response: str) -> dict[str, list[float]]:
		"""Parse a raw comma-separated source/current/time string."""
		results = {"Source": [], "Current": [], "Time": []}
		split_arr = [item.strip() for item in raw_response.split(",") if item.strip()]

		for index, value in enumerate(split_arr):
			numeric = float(value)
			if np.mod(index, 3) == 0:
				results["Source"].append(numeric)
			elif np.mod(index, 3) == 1:
				results["Current"].append(numeric)
			else:
				results["Time"].append(numeric)

		return results

	def simple_IV_sweep(
		self,
		vstart: float,
		vstop: float,
		vstep: float,
		compliance: float,
		delay: float,
		adr: str | None = None,
	) -> np.ndarray:
		"""Run a simple linear IV sweep using the universal driver primitive."""
		if vstep == 0:
			raise ValueError("vstep must be non-zero.")

		stop_correction = np.sign(vstep) * 0.5 * abs(vstep)
		voltages = np.arange(vstart, vstop + stop_correction, vstep, dtype=float).tolist()

		records = self.driver.source_voltage_measure_current(
			voltages=voltages,
			current_compliance=float(compliance),
			delay_s=float(delay),
			reset=True,
			adr=adr,
		)
		compatibility_records = self._driver_records_to_compatibility(records)
		return _records_to_legacy_array(compatibility_records)

	def list_IV_sweep_manual(
		self,
		csv_path: str,
		pos_compliance: float,
		neg_compliance: float,
		delay: float | None = None,
		adr: str | None = None,
	) -> np.ndarray:
		"""Run a manual point-by-point voltage list using driver primitives."""
		voltages, times = self._coerce_voltage_list(csv_path=csv_path, bare_list=None)
		resolved_delay, _ = self._resolve_delay_and_acq(times, delay, None)

		records: list[dict[str, float | None]] = []
		start_time = time.perf_counter()

		for index, voltage in enumerate(voltages):
			compliance = pos_compliance if voltage >= 0 else neg_compliance
			point_records = self.driver.hold_voltage_measure_current(
				voltage=float(voltage),
				current_compliance=float(compliance),
				settle_s=max(0.0, resolved_delay),
				read_count=1,
				reset=(index == 0),
				adr=adr,
			)
			point = self._driver_records_to_compatibility(point_records)[0]
			point["Time(T)"] = time.perf_counter() - start_time
			records.append(point)

		return _records_to_legacy_array(records)

	def scan_read_vlist(
		self,
		dev: Any,
		voltage_list: Sequence[float],
		set_width: float,
		set_acquire_delay: float,
		current_compliance: float,
		set_range: float | None = None,
	) -> list[dict[str, float | None]]:
		"""Run one voltage list against the wrapped driver.

		The ``dev`` argument is kept for backward compatibility with older call
		sites. The active wrapped driver is always used.
		"""
		_ = dev
		if self.supports_native_pulse:
			driver_records = self.driver.run_voltage_pulse_train(
				voltages=[float(value) for value in voltage_list],
				current_compliance=float(current_compliance),
				pulse_width_s=float(set_width),
				acquire_delay_s=float(set_acquire_delay),
				current_range=set_range,
				reset=True,
			)
		else:
			driver_records = self.driver.run_voltage_pulse_train(
				voltages=[float(value) for value in voltage_list],
				current_compliance=float(current_compliance),
				pulse_width_s=float(set_width),
				acquire_delay_s=float(set_acquire_delay),
				current_range=set_range,
				reset=True,
			)

		return self._driver_records_to_compatibility(driver_records)

	def _default_read_probe_list(self) -> list[float]:
		"""Return the configured tiny-IV read-probe list."""
		if self.resistance_df is None:
			raise ValueError("No tiny-IV CSV has been configured for read probes.")
		return self.resistance_df.iloc[:, 1].astype(float).tolist()

	def list_IV_sweep_split(
		self,
		csv_path: str,
		pos_compliance: float,
		neg_compliance: float,
		SMU_range: float | None = None,
		delay: float | None = None,
		acq_delay: float | None = None,
		adr: str | None = None,
		pos_channel: Any | None = None,
		neg_channel: Any | None = None,
		include_read_probe: bool = True,
		read_probe_mode: str = "between_segments",
	) -> tuple[list[dict[str, float | None]], list[dict[str, float | None]]]:
		"""Split a list into positive and negative segments and run them sequentially."""
		_ = pos_channel
		_ = neg_channel
		voltages, times = self._coerce_voltage_list(csv_path=csv_path, bare_list=None)
		delay_value, acq_value = self._resolve_delay_and_acq(times, delay, acq_delay)
		splits = self.split_list(voltages)

		data_array: list[dict[str, float | None]] = []
		resistance_array: list[dict[str, float | None]] = []

		for cycle_number, tag, segment in splits:
			compliance = float(pos_compliance if tag == "p" else neg_compliance)
			segment_records = self.scan_read_vlist(
				dev=self.driver,
				voltage_list=segment,
				set_width=delay_value,
				set_acquire_delay=acq_value,
				current_compliance=compliance,
				set_range=SMU_range,
			)
			for record in segment_records:
				record["Cycle Number"] = float(cycle_number)
			data_array.extend(segment_records)

			if include_read_probe and read_probe_mode == "between_segments":
				read_probe_list = self._default_read_probe_list()
				res_compliance = 10e-6 if tag == "p" else 10e-3
				res_range = 10e-8 if tag == "p" else 10e-4
				probe_records = self.scan_read_vlist(
					dev=self.driver,
					voltage_list=read_probe_list,
					set_width=10e-4,
					set_acquire_delay=5e-4,
					current_compliance=res_compliance,
					set_range=res_range,
				)
				for record in probe_records:
					record["Cycle Number"] = float(cycle_number)
				resistance_array.extend(probe_records)

		return data_array, resistance_array

	def list_IV_sweep_split_4(
		self,
		csv_path: str,
		pos_compliance: float | None = None,
		neg_compliance: float | None = None,
		SMU_range: float | None = None,
		delay: float | None = None,
		acq_delay: float | None = None,
		adr: str | None = None,
		pos_channel: Any | None = None,
		neg_channel: Any | None = None,
		include_read_probe: bool = True,
		compliance_pf: float | None = None,
		compliance_pb: float | None = None,
		compliance_nf: float | None = None,
		compliance_nb: float | None = None,
		wait_time: float | None = None,
		progress_callback: Any | None = None,
		read_probe_mode: str = "between_segments",
	) -> tuple[list[dict[str, float | None]], list[dict[str, float | None]]]:
		"""Split a list into four directional segments and run them sequentially."""
		_ = adr
		_ = pos_channel
		_ = neg_channel

		voltages, times = self._coerce_voltage_list(csv_path=csv_path, bare_list=None)
		delay_value, acq_value = self._resolve_delay_and_acq(times, delay, acq_delay)
		splits = self.split_list_by_4(voltages)

		if pos_compliance is None:
			pos_compliance = compliance_pf if compliance_pf is not None else compliance_pb
		if neg_compliance is None:
			neg_compliance = compliance_nf if compliance_nf is not None else compliance_nb
		if pos_compliance is None or neg_compliance is None:
			raise ValueError(
				"Provide pos/neg compliance or all four segment-specific values."
			)

		compliance_map = {
			"pf": float(compliance_pf if compliance_pf is not None else pos_compliance),
			"pb": float(compliance_pb if compliance_pb is not None else pos_compliance),
			"nf": float(compliance_nf if compliance_nf is not None else neg_compliance),
			"nb": float(compliance_nb if compliance_nb is not None else neg_compliance),
		}

		data_array: list[dict[str, float | None]] = []
		resistance_array: list[dict[str, float | None]] = []

		for segment_index, (cycle_number, tag, segment) in enumerate(splits):
			if wait_time is not None and wait_time > 0:
				time.sleep(wait_time)

			segment_records = self.scan_read_vlist(
				dev=self.driver,
				voltage_list=segment,
				set_width=delay_value,
				set_acquire_delay=acq_value,
				current_compliance=compliance_map[tag],
				set_range=SMU_range,
			)
			for record in segment_records:
				record["Cycle Number"] = float(cycle_number)
			data_array.extend(segment_records)

			if callable(progress_callback):
				progress_callback(segment_index + 1, len(splits))

			if (
				include_read_probe
				and read_probe_mode == "between_segments"
				and tag in {"pb", "nb"}
			):
				read_probe_list = self._default_read_probe_list()
				res_compliance = 10e-3 if tag == "pb" else 10e-6
				res_range = 10e-4 if tag == "pb" else 10e-8
				probe_records = self.scan_read_vlist(
					dev=self.driver,
					voltage_list=read_probe_list,
					set_width=10e-4,
					set_acquire_delay=5e-4,
					current_compliance=res_compliance,
					set_range=res_range,
				)
				for record in probe_records:
					record["Cycle Number"] = float(cycle_number)
				resistance_array.extend(probe_records)

		return data_array, resistance_array

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
		"""Run a pulse train using the best primitive for the active driver."""
		_ = adr
		_ = current_autorange

		voltage_list, times = self._coerce_voltage_list(csv_path=csv_path, bare_list=bare_list)

		if current_compliance is None:
			current_compliance = compliance
		if current_compliance is None:
			raise ValueError("A current compliance value is required.")

		if pulse_width is not None:
			set_width = pulse_width

		if set_acquire_delay is None:
			_, set_acquire_delay = self._resolve_delay_and_acq(times, set_width, None)

		driver_records = self.driver.run_voltage_pulse_train(
			voltages=voltage_list,
			current_compliance=float(current_compliance),
			pulse_width_s=float(set_width),
			acquire_delay_s=float(set_acquire_delay),
			current_range=None,
			reset=True,
		)
		return self._driver_records_to_compatibility(driver_records)

	def general_channel_pulsing(
		self,
		adr: str | None = None,
		measurement_type: str = "single",
		mode: int = 1,
		positive_voltages: Sequence[float] | None = None,
		negative_voltages: Sequence[float] | None = None,
		set_width: float = 50e-3,
		set_acquire_delay: float | None = None,
		current_compliance: float = 1e-5,
		set_range: float = 1e-5,
	) -> np.ndarray:
		"""Backward-compatible pulse helper.

		For single-channel mode this forwards to ``scan_read_vlist``. For the
		legacy two-channel signed-voltage approximation, positive and negative
		lists are merged algebraically before execution. This keeps the interface
		stable while leaving the true low-level SCPI work in the driver.
		"""
		_ = adr
		_ = mode

		if positive_voltages is None:
			raise ValueError("positive_voltages is required.")

		if set_acquire_delay is None:
			set_acquire_delay = float(set_width) / 2

		if measurement_type == "single":
			records = self.scan_read_vlist(
				dev=self.driver,
				voltage_list=positive_voltages,
				set_width=set_width,
				set_acquire_delay=set_acquire_delay,
				current_compliance=current_compliance,
				set_range=set_range,
			)
			return _records_to_legacy_array(records)

		if negative_voltages is None:
			raise ValueError("negative_voltages is required in double mode.")

		if len(positive_voltages) != len(negative_voltages):
			raise ValueError("Positive and negative voltage lists must have the same length.")

		combined = [
			float(vp) - float(vn)
			for vp, vn in zip(positive_voltages, negative_voltages)
		]
		records = self.scan_read_vlist(
			dev=self.driver,
			voltage_list=combined,
			set_width=set_width,
			set_acquire_delay=set_acquire_delay,
			current_compliance=current_compliance,
			set_range=set_range,
		)
		return _records_to_legacy_array(records)

	def get_frame_from_records(
		self,
		records: list[dict[str, float | None]],
	) -> pd.DataFrame:
		"""Return a DataFrame from compatibility records."""
		return _records_to_frame(records)


class KeithleySMU(SourceMeasureUnit):
	"""Backward-compatible GUI-facing Keithley wrapper."""

	def __init__(
		self,
		device_no: int = 0,
		address: str | None = None,
		switch: Any | None = None,
		switch_channel: Any | None = None,
		connect_switch: bool = False,
		tiny_iv_path: str | None = None,
	) -> None:
		"""Create the universal layer around a Keithley 2450 driver."""
		_ = connect_switch
		driver = Keithley2450(
			device_no=device_no,
			address=address,
			switch=switch,
			switch_channel=switch_channel,
			connect_switch=False,
		)
		super().__init__(driver=driver, tiny_iv_path=tiny_iv_path)


class KeysightSMU(SourceMeasureUnit):
	"""Backward-compatible GUI-facing Keysight wrapper."""

	def __init__(
		self,
		device_no: int = 0,
		address: str | None = None,
		switch: Any | None = None,
		switch_channel: Any | None = None,
		connect_switch: bool = False,
		tiny_iv_path: str | None = None,
		channel: int = 1,
	) -> None:
		"""Create the universal layer around a Keysight B2902B driver."""
		_ = connect_switch
		driver = KeysightB2902B(
			device_no=device_no,
			address=address,
			switch=switch,
			switch_channel=switch_channel,
			connect_switch=False,
			channel=channel,
		)
		super().__init__(driver=driver, tiny_iv_path=tiny_iv_path)
