"""Shared SCPI driver skeletons for ViPSA source-measure units.

This module intentionally contains no instrument-control implementation. It is
a contract and documentation layer for the Keithley and Keysight driver files.
Fill in each method body with the SCPI sequence appropriate for the target
instrument while keeping the public signatures stable for the future
``Source_Measure_Unit.py`` handler layer.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

try:
    import pyvisa
except ModuleNotFoundError:
    pyvisa = types.SimpleNamespace(ResourceManager=None)
    sys.modules.setdefault("pyvisa", pyvisa)


class ScpiError(RuntimeError):
    """Base exception type for SCPI driver errors."""


class ScpiConnectionError(ScpiError):
    """Raised when a VISA resource cannot be opened, used, or closed."""


class ScpiResponseError(ScpiError):
    """Raised when an instrument response cannot be parsed into expected data."""


@dataclass(slots=True)
class ScpiReading:
    """Container for one normalized source-measure reading.

    Intended fields:
        timestamp_s: Relative timestamp of the measurement in seconds.
        voltage_v: Measured voltage in volts, when available.
        current_a: Measured current in amperes, when available.
        resistance_ohm: Derived or measured resistance in ohms, when available.
        source_value: Commanded source value for the point.
        raw: Raw instrument response or diagnostic text associated with the point.
    """

    timestamp_s: float = 0.0
    voltage_v: float | None = None
    current_a: float | None = None
    resistance_ohm: float | None = None
    source_value: float | None = None
    raw: str = ""

    def as_dict(self) -> dict[str, float | str | None]:
        """Return this reading as a plain dictionary.

        Implement this if callers need serialization without depending on the
        dataclass API. The returned keys should match the dataclass field names.
        """
        return {}


def normalize_function(function: str) -> str:
    """Normalize user-facing function names into SCPI function tokens.

    Intended behavior:
        Accept aliases such as ``"voltage"``, ``"volt"``, ``"current"``,
        ``"curr"``, ``"resistance"``, ``"source"``, and ``"time"``.
        Return the short SCPI token used by the vendor drivers, for example
        ``"VOLT"``, ``"CURR"``, ``"RES"``, ``"SOUR"``, or ``"TIME"``.
    """
    return ""


def normalize_functions(functions: str | Sequence[str]) -> list[str]:
    """Normalize one or more function names into SCPI format tokens.

    Intended behavior:
        Support either a single string or a sequence of strings, preserving
        order for format commands such as ``VOLT,CURR,TIME``.
    """
    return []


def parse_numeric_csv(response: Any, *, expected_min: int = 1) -> list[float]:
    """Parse a comma-separated ASCII SCPI numeric response.

    Intended behavior:
        Convert fields such as ``"0.1,2e-6,0.001"`` into floats, reject
        malformed values with ``ScpiResponseError``, and verify that at least
        ``expected_min`` numbers were present.
    """
    return []


def unsupported_compound_method(method_name: str) -> NotImplementedError:
    """Build the error used by deferred orchestration-level APIs.

    Intended behavior:
        Return a ``NotImplementedError`` explaining that CSV loading, sweep
        segmentation, read-probe insertion, and metadata handling belong in the
        future ``Source_Measure_Unit.py`` handler rather than these thin
        instrument drivers.
    """
    return NotImplementedError(method_name)


class SCPIInstrument:
    """Abstract skeleton for a minimal VISA-backed SCPI instrument.

    Subclasses should translate this generic surface into vendor-specific SCPI
    commands. The base class documents lifecycle, command, configuration, and
    baseline measurement operations but intentionally performs no I/O yet.
    """

    read_termination = "\n"
    write_termination = "\n"

    def __init__(
        self,
        device_no: int = 0,
        address: str | None = None,
        timeout: int = 10000,
        auto_connect: bool = True,
    ) -> None:
        """Store constructor parameters for a future VISA session.

        Intended behavior:
            Record the selected VISA device index, explicit address, timeout,
            and whether construction should immediately open the instrument.
            A concrete implementation may create a ``pyvisa.ResourceManager``
            and session here or defer that work to ``connect``.
        """
        return None

    def connect(self, address: str | None = None, device_no: int = 0) -> "SCPIInstrument":
        """Open and configure a VISA session.

        Intended behavior:
            Resolve the target resource from ``address`` or ``device_no``, open
            it with PyVISA, apply timeout and read/write termination settings,
            and retain the session for later ``write`` and ``query`` calls.
        """
        return self

    def disconnect(self) -> None:
        """Close the active VISA session and resource manager.

        Intended behavior:
            Turn outputs off if appropriate, close the instrument session, close
            the resource manager, and leave the object in a reconnectable state.
        """
        return None

    def write(self, command: str) -> Any:
        """Send one SCPI command without reading a response.

        Intended behavior:
            Validate that a session exists, send ``command`` exactly as supplied,
            and return the underlying PyVISA write result if useful.
        """
        return None

    def query(self, command: str) -> Any:
        """Send one SCPI query and return the raw instrument response.

        Intended behavior:
            Validate that a session exists, send ``command``, read the response,
            and return it without instrument-specific parsing.
        """
        return None

    def ask(self, command: str) -> Any:
        """Backward-compatible alias for ``query``.

        Intended behavior:
            Preserve older call sites that use PyVISA's historical ``ask`` name.
        """
        return None

    def clear(self) -> "SCPIInstrument":
        """Clear the instrument status/error queues.

        Intended behavior:
            Send the SCPI clear command, typically ``*CLS``, and return ``self``
            for fluent setup sequences.
        """
        return self

    def reset(self) -> "SCPIInstrument":
        """Reset the instrument to a known baseline state.

        Intended behavior:
            Send the SCPI reset command, typically ``*RST``, clear pending status
            events, and return ``self``.
        """
        return self

    def initialize(self) -> "SCPIInstrument":
        """Apply the default ViPSA source-measure state.

        Intended behavior:
            Configure a conservative baseline such as source voltage, sense
            current/voltage, ASCII readback format, safe ranges, and output off.
        """
        return self

    def sync(self) -> Any:
        """Block until the current instrument operation completes.

        Intended behavior:
            Use ``*OPC?`` or the vendor-specific equivalent and return the raw
            completion response.
        """
        return None

    def read_errors(self) -> list[str]:
        """Read queued instrument errors.

        Intended behavior:
            Drain the instrument error queue and return all error messages except
            the final no-error sentinel.
        """
        return []

    def get_address(self) -> str | None:
        """Return the currently configured VISA address."""
        return None

    def set_source_function(self, function: str) -> None:
        """Select the active source function.

        Intended behavior:
            Map ``function`` to the vendor's voltage/current source command and
            send the relevant SCPI setup instruction.
        """
        return None

    def set_sense_function(self, function: str | Sequence[str]) -> None:
        """Select one or more measurement functions.

        Intended behavior:
            Enable current, voltage, resistance, or combined readback functions
            in the order expected by subsequent fetch/read parsing.
        """
        return None

    def set_source_range(self, function: str, value: float | None = None, auto: bool = True) -> None:
        """Configure the source range for voltage or current sourcing.

        Intended behavior:
            Use autorange when ``auto`` is true; otherwise apply the fixed range
            in ``value`` for the selected source function.
        """
        return None

    def set_sense_range(self, function: str, value: float | None = None, auto: bool = True) -> None:
        """Configure the measurement range for voltage/current/resistance.

        Intended behavior:
            Use autorange when requested, or apply the fixed measurement range
            needed for low-noise/current-limited measurements.
        """
        return None

    def set_protection(self, function: str, value: float) -> None:
        """Configure voltage or current protection/compliance.

        Intended behavior:
            Apply current compliance while sourcing voltage, or voltage limit
            while sourcing current, using the vendor-specific SCPI command.
        """
        return None

    def set_source_level(self, function: str, value: float) -> None:
        """Set the immediate source level for voltage or current."""
        return None

    def set_output(self, enabled: bool) -> None:
        """Turn the SMU output on or off."""
        return None

    def abort(self) -> None:
        """Abort an in-progress measurement and leave the output safe."""
        return None

    def configure_measurement_format(self, functions: str | Sequence[str]) -> None:
        """Configure the order and type of values returned by reads/fetches."""
        return None

    def query_reading(self) -> str:
        """Read one measurement response from the active configuration."""
        return ""

    def parse_voltage_current(self, response: Any) -> tuple[float | None, float | None]:
        """Parse a response containing measured voltage and current fields."""
        return None, None

    def source_voltage_measure_current(
        self,
        voltages: Iterable[float],
        current_compliance: float,
        delay_s: float = 0.0,
        current_range: float | None = None,
        voltage_range: float | None = None,
        reset: bool = True,
    ) -> list[ScpiReading]:
        """Source voltage points and measure current at each point.

        Intended behavior:
            Configure voltage sourcing, current measurement, compliance, ranges,
            timing, and readback format. Apply every commanded voltage in order
            and return one ``ScpiReading`` per point.
        """
        return []

    def source_current_measure_voltage(
        self,
        currents: Iterable[float],
        voltage_limit: float,
        delay_s: float = 0.0,
        voltage_range: float | None = None,
        current_range: float | None = None,
        reset: bool = True,
    ) -> list[ScpiReading]:
        """Source current points and measure voltage at each point.

        Intended behavior:
            Configure current sourcing, voltage measurement, voltage limit,
            ranges, timing, and readback format. Return one reading per current.
        """
        return []

    def read_current_at_voltage(
        self,
        voltage: float,
        current_compliance: float,
        settle_s: float = 0.02,
        current_range: float | None = None,
        reset: bool = True,
    ) -> ScpiReading:
        """Source one DC voltage and return the measured current.

        Intended behavior:
            This is the constant-voltage/current-read helper used for contact
            checks and simple resistance probes.
        """
        return ScpiReading()

    def measure_resistance(
        self,
        test_current: float = 1e-6,
        voltage_limit: float = 1.0,
        settle_s: float = 0.02,
        reset: bool = True,
    ) -> ScpiReading:
        """Measure resistance using a minimal source-current/read-voltage flow."""
        return ScpiReading()

    def run_voltage_pulse_train(
        self,
        voltages: Sequence[float],
        current_compliance: float,
        pulse_width_s: float,
        acquire_delay_s: float | None = None,
        current_range: float | None = None,
        reset: bool = True,
    ) -> list[ScpiReading]:
        """Run or approximate a voltage pulse train and return measured points."""
        return []
