from __future__ import annotations

from typing import List, Tuple


EPSILON = 1e-9
SWEEP_MODE_POSITIVE_FIRST = "positive_first"
SWEEP_MODE_NEGATIVE_FIRST = "negative_first"
SWEEP_MODE_POSITIVE_TO_NEGATIVE = "positive_to_negative"
SWEEP_MODE_NEGATIVE_TO_POSITIVE = "negative_to_positive"
SWEEP_MODE_OPTIONS = (
    SWEEP_MODE_POSITIVE_FIRST,
    SWEEP_MODE_NEGATIVE_FIRST,
    SWEEP_MODE_POSITIVE_TO_NEGATIVE,
    SWEEP_MODE_NEGATIVE_TO_POSITIVE,
)


def inclusive_ramp(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("Step voltage must be greater than zero.")

    direction = 1 if end >= start else -1
    actual_step = abs(step) * direction
    values: List[float] = []
    current = float(start)

    while True:
        values.append(round(current, 12))
        current += actual_step
        if direction > 0 and current > end + EPSILON:
            break
        if direction < 0 and current < end - EPSILON:
            break

    if abs(values[-1] - float(end)) > EPSILON:
        values.append(round(float(end), 12))

    return values


def add_hold_points(sequence: List[float], hold_value: float, hold_steps: int) -> List[float]:
    if hold_steps <= 0:
        return list(sequence)
    return list(sequence) + [round(float(hold_value), 12)] * int(hold_steps)


def _positive_leg(forward_voltage: float, step_voltage: float, peak_hold_steps: int) -> List[float]:
    up = inclusive_ramp(0.0, forward_voltage, step_voltage)
    down = up[-2::-1] if len(up) > 1 else []
    return add_hold_points(up, forward_voltage, peak_hold_steps) + down


def _negative_leg(reset_voltage: float, step_voltage: float, peak_hold_steps: int) -> List[float]:
    down = inclusive_ramp(0.0, reset_voltage, step_voltage)
    up = down[-2::-1] if len(down) > 1 else []
    return add_hold_points(down, reset_voltage, peak_hold_steps) + up


def build_sweep_cycle(
    forward_voltage: float,
    reset_voltage: float,
    step_voltage: float,
    peak_hold_steps: int = 0,
    sweep_mode: str = SWEEP_MODE_POSITIVE_FIRST,
) -> List[float]:
    mode = str(sweep_mode or SWEEP_MODE_POSITIVE_FIRST).strip().lower()
    if mode not in SWEEP_MODE_OPTIONS:
        raise ValueError(
            f"Unsupported sweep_mode '{sweep_mode}'. Expected one of {', '.join(SWEEP_MODE_OPTIONS)}."
        )

    if mode == SWEEP_MODE_POSITIVE_FIRST:
        return _positive_leg(forward_voltage, step_voltage, peak_hold_steps) + _negative_leg(
            reset_voltage, step_voltage, peak_hold_steps
        )

    if mode == SWEEP_MODE_NEGATIVE_FIRST:
        return _negative_leg(reset_voltage, step_voltage, peak_hold_steps) + _positive_leg(
            forward_voltage, step_voltage, peak_hold_steps
        )

    if mode == SWEEP_MODE_POSITIVE_TO_NEGATIVE:
        return inclusive_ramp(forward_voltage, reset_voltage, step_voltage)

    return inclusive_ramp(reset_voltage, forward_voltage, step_voltage)


def infer_cycle_numbers(voltages: List[float]) -> List[int]:
    if not voltages:
        return []

    start_marker = None
    for value in voltages:
        if abs(float(value)) > EPSILON:
            start_marker = float(value)
            break

    if start_marker is None:
        return [1] * len(voltages)

    cycle_numbers: List[int] = []
    current_cycle = 1
    seen_non_zero = False
    previous_voltage = 0.0

    for value in voltages:
        voltage = float(value)
        if (
            abs(previous_voltage) <= EPSILON
            and abs(voltage - start_marker) <= EPSILON
            and seen_non_zero
        ):
            current_cycle += 1
            seen_non_zero = False

        cycle_numbers.append(current_cycle)
        if abs(voltage) > EPSILON:
            seen_non_zero = True
        previous_voltage = voltage

    return cycle_numbers


def generate_voltage_data(
    forward_voltage: float,
    reset_voltage: float,
    step_voltage: float,
    timer_delay: float,
    forming_cycle: bool,
    forming_voltage: float | None,
    cycles: int,
    peak_hold_steps: int = 0,
    return_to_zero: bool = True,
    sweep_mode: str = SWEEP_MODE_POSITIVE_FIRST,
) -> Tuple[List[float], List[float], List[int]]:
    voltages: List[float] = []
    times: List[float] = []
    current_time = 0.0

    def append_sequence(sequence: List[float]) -> None:
        nonlocal current_time
        for voltage in sequence:
            voltages.append(voltage)
            times.append(round(current_time, 12))
            current_time += timer_delay

    if forming_cycle:
        if forming_voltage is None:
            raise ValueError("Forming voltage is required when forming_cycle is enabled.")
        append_sequence(
            build_sweep_cycle(
                forming_voltage,
                reset_voltage,
                step_voltage,
                peak_hold_steps=peak_hold_steps,
                sweep_mode=sweep_mode,
            )
        )

    for _ in range(int(cycles)):
        append_sequence(
            build_sweep_cycle(
                forward_voltage,
                reset_voltage,
                step_voltage,
                peak_hold_steps=peak_hold_steps,
                sweep_mode=sweep_mode,
            )
        )

    if return_to_zero and voltages and abs(voltages[-1]) > EPSILON:
        voltages.append(0.0)
        times.append(round(current_time, 12))

    return times, voltages, infer_cycle_numbers(voltages)
