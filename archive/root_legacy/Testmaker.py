"""
Standalone test-maker app for ViPSA.

This app stays separate from the current protocol handling. It generates
protocol-compatible DC sweep and pulse CSV files and can export JSON protocol
files in the current {"type": ..., "params": ...} format.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

try:
    import PySimpleGUI as sg
except ModuleNotFoundError:
    sg = None


STATUS_READY = "Ready Now"
STATUS_PARTIAL = "Partial / Manual"
STATUS_NOT_READY = "Not Currently Feasible"

MODE_SWEEP = "sweep"
MODE_PULSE = "pulse"
MODE_INFO = "info"

EPSILON = 1e-9

TEST_CATALOG: List[Dict[str, object]] = [
    {
        "name": "Current-Voltage (I-V) Sweep",
        "category": "Basic Electrical",
        "status": STATUS_READY,
        "mode": MODE_SWEEP,
        "generator": "dciv_bipolar",
        "protocol_type": "DCIV",
        "description": "Standard bipolar sweep for switching curves and VSET/VRESET extraction.",
        "feasibility": "Directly supported by the current DCIV workflow.",
        "defaults": {
            "forward_voltage": 1.5,
            "reset_voltage": -1.5,
            "step_voltage": 0.05,
            "timer_delay": 0.001,
            "cycles": 2,
            "forming_cycle": False,
            "forming_voltage": 2.0,
            "peak_hold_steps": 0,
            "return_to_zero": True,
            "pos_compl": 0.001,
            "neg_compl": 0.01,
            "use_4way_split": True,
        },
    },
    {
        "name": "Forming / Electroforming Sweep",
        "category": "Basic Electrical",
        "status": STATUS_READY,
        "mode": MODE_SWEEP,
        "generator": "dciv_bipolar",
        "protocol_type": "DCIV",
        "description": "A higher first sweep followed by regular bipolar sweeps.",
        "feasibility": "Supported directly by the sweep generator and DCIV step.",
        "defaults": {
            "forward_voltage": 1.5,
            "reset_voltage": -1.5,
            "step_voltage": 0.05,
            "timer_delay": 0.001,
            "cycles": 2,
            "forming_cycle": True,
            "forming_voltage": 2.5,
            "peak_hold_steps": 0,
            "return_to_zero": True,
            "pos_compl": 0.001,
            "neg_compl": 0.01,
            "use_4way_split": True,
        },
    },
    {
        "name": "Read Voltage Check",
        "category": "Basic Electrical",
        "status": STATUS_READY,
        "mode": MODE_PULSE,
        "generator": "retention_read",
        "protocol_type": "PULSE",
        "description": "Low-voltage read train for checking non-destructive sensing.",
        "feasibility": "Supported as a protocol-compatible pulse list.",
        "defaults": {
            "base_width": 0.001,
            "write_voltage": 0.8,
            "write_width": 0.002,
            "write_gap": 0.001,
            "write_pulses": 1,
            "read_voltage": 0.1,
            "read_width": 0.001,
            "read_gap": 0.05,
            "read_pulses": 20,
            "erase_voltage": -0.8,
            "erase_width": 0.002,
            "erase_gap": 0.001,
            "erase_pulses": 0,
            "pulse_cycles": 1,
            "cycle_gap": 0.0,
            "initial_gap": 0.0,
            "final_read_block": False,
            "pulse_compliance": 0.001,
            "set_acquire_delay": 0.0005,
        },
    },
    {
        "name": "Endurance Cycling",
        "category": "Memory Performance",
        "status": STATUS_READY,
        "mode": MODE_PULSE,
        "generator": "endurance",
        "protocol_type": "PULSE",
        "description": "Repeated write-read-erase-read cycles.",
        "feasibility": "Fully compatible with the current pulse execution model.",
        "defaults": {
            "base_width": 0.001,
            "write_voltage": 1.5,
            "write_width": 0.001,
            "write_gap": 0.001,
            "write_pulses": 1,
            "read_voltage": 0.1,
            "read_width": 0.001,
            "read_gap": 0.001,
            "read_pulses": 1,
            "erase_voltage": -1.8,
            "erase_width": 0.001,
            "erase_gap": 0.001,
            "erase_pulses": 1,
            "pulse_cycles": 100,
            "cycle_gap": 0.0,
            "initial_gap": 0.0,
            "final_read_block": True,
            "pulse_compliance": 0.01,
            "set_acquire_delay": 0.0006,
        },
    },
    {
        "name": "Retention Readout Train",
        "category": "Memory Performance",
        "status": STATUS_PARTIAL,
        "mode": MODE_PULSE,
        "generator": "retention_read",
        "protocol_type": "PULSE",
        "description": "Program once, then read repeatedly after zero-voltage gaps.",
        "feasibility": "Works for modest intervals. Very long waits create very large slot lists.",
        "defaults": {
            "base_width": 0.01,
            "write_voltage": 1.2,
            "write_width": 0.01,
            "write_gap": 0.01,
            "write_pulses": 1,
            "read_voltage": 0.1,
            "read_width": 0.01,
            "read_gap": 1.0,
            "read_pulses": 30,
            "erase_voltage": -1.0,
            "erase_width": 0.01,
            "erase_gap": 0.01,
            "erase_pulses": 0,
            "pulse_cycles": 1,
            "cycle_gap": 0.0,
            "initial_gap": 0.0,
            "final_read_block": False,
            "pulse_compliance": 0.001,
            "set_acquire_delay": 0.005,
        },
    },
    {
        "name": "Cycle-to-Cycle Variability",
        "category": "Memory Performance",
        "status": STATUS_PARTIAL,
        "mode": MODE_PULSE,
        "generator": "endurance",
        "protocol_type": "PULSE",
        "description": "Use repeated switching cycles to collect distributions on one device.",
        "feasibility": "Acquisition is feasible now; the statistics are still manual.",
        "defaults": {
            "base_width": 0.001,
            "write_voltage": 1.5,
            "write_width": 0.001,
            "write_gap": 0.001,
            "write_pulses": 1,
            "read_voltage": 0.1,
            "read_width": 0.001,
            "read_gap": 0.001,
            "read_pulses": 1,
            "erase_voltage": -1.8,
            "erase_width": 0.001,
            "erase_gap": 0.001,
            "erase_pulses": 1,
            "pulse_cycles": 200,
            "cycle_gap": 0.0,
            "initial_gap": 0.0,
            "final_read_block": True,
            "pulse_compliance": 0.01,
            "set_acquire_delay": 0.0006,
        },
    },
    {
        "name": "Device-to-Device Variability",
        "category": "Memory Performance",
        "status": STATUS_PARTIAL,
        "mode": MODE_SWEEP,
        "generator": "dciv_bipolar",
        "protocol_type": "DCIV",
        "description": "Run the same sweep across many devices and compare distributions.",
        "feasibility": "The list is supported now; multi-device analysis remains external.",
        "defaults": {
            "forward_voltage": 1.2,
            "reset_voltage": -1.2,
            "step_voltage": 0.02,
            "timer_delay": 0.001,
            "cycles": 2,
            "forming_cycle": False,
            "forming_voltage": 2.0,
            "peak_hold_steps": 0,
            "return_to_zero": True,
            "pos_compl": 0.001,
            "neg_compl": 0.01,
            "use_4way_split": True,
        },
    },
    {
        "name": "Multi-Level Cell Programming",
        "category": "Memory Performance",
        "status": STATUS_PARTIAL,
        "mode": MODE_PULSE,
        "generator": "ltp_ltd",
        "protocol_type": "PULSE",
        "description": "Incremental conductance stepping with repeated pot/depot pulses.",
        "feasibility": "Generation is supported; closed-loop level targeting is not yet automated.",
        "defaults": {
            "base_width": 0.001,
            "write_voltage": 1.0,
            "write_width": 0.001,
            "write_gap": 0.001,
            "write_pulses": 20,
            "read_voltage": 0.1,
            "read_width": 0.001,
            "read_gap": 0.001,
            "read_pulses": 1,
            "erase_voltage": -1.0,
            "erase_width": 0.001,
            "erase_gap": 0.001,
            "erase_pulses": 20,
            "pulse_cycles": 1,
            "cycle_gap": 0.0,
            "initial_gap": 0.0,
            "final_read_block": False,
            "pulse_compliance": 0.005,
            "set_acquire_delay": 0.0006,
        },
    },
    {
        "name": "Bias Stress / Operational Fatigue",
        "category": "Environmental & Stability",
        "status": STATUS_READY,
        "mode": MODE_PULSE,
        "generator": "bias_stress",
        "protocol_type": "PULSE",
        "description": "Repeated fixed-bias stress pulses with optional read probes.",
        "feasibility": "Supported as a fixed-width pulse-slot sequence.",
        "defaults": {
            "base_width": 0.001,
            "write_voltage": 0.8,
            "write_width": 0.1,
            "write_gap": 0.01,
            "write_pulses": 1,
            "read_voltage": 0.1,
            "read_width": 0.001,
            "read_gap": 0.01,
            "read_pulses": 1,
            "erase_voltage": 0.0,
            "erase_width": 0.001,
            "erase_gap": 0.0,
            "erase_pulses": 0,
            "pulse_cycles": 50,
            "cycle_gap": 0.1,
            "initial_gap": 0.0,
            "final_read_block": False,
            "pulse_compliance": 0.005,
            "set_acquire_delay": 0.0006,
        },
    },
    {
        "name": "Temperature Stability",
        "category": "Environmental & Stability",
        "status": STATUS_PARTIAL,
        "mode": MODE_SWEEP,
        "generator": "dciv_bipolar",
        "protocol_type": "DCIV",
        "description": "Reuse the same electrical list while an external heater controls temperature.",
        "feasibility": "Electrical lists are feasible; temperature control is external to this codebase.",
        "defaults": {
            "forward_voltage": 1.0,
            "reset_voltage": -1.0,
            "step_voltage": 0.02,
            "timer_delay": 0.001,
            "cycles": 1,
            "forming_cycle": False,
            "forming_voltage": 2.0,
            "peak_hold_steps": 0,
            "return_to_zero": True,
            "pos_compl": 0.001,
            "neg_compl": 0.01,
            "use_4way_split": True,
        },
    },
    {
        "name": "Humidity Stability",
        "category": "Environmental & Stability",
        "status": STATUS_PARTIAL,
        "mode": MODE_SWEEP,
        "generator": "dciv_bipolar",
        "protocol_type": "DCIV",
        "description": "Reuse the same electrical list while an external RH setup controls humidity.",
        "feasibility": "Electrical lists are feasible; humidity control is not integrated.",
        "defaults": {
            "forward_voltage": 1.0,
            "reset_voltage": -1.0,
            "step_voltage": 0.02,
            "timer_delay": 0.001,
            "cycles": 1,
            "forming_cycle": False,
            "forming_voltage": 2.0,
            "peak_hold_steps": 0,
            "return_to_zero": True,
            "pos_compl": 0.001,
            "neg_compl": 0.01,
            "use_4way_split": True,
        },
    },
    {
        "name": "Bending / Flexibility",
        "category": "Environmental & Stability",
        "status": STATUS_NOT_READY,
        "mode": MODE_INFO,
        "generator": None,
        "protocol_type": None,
        "description": "Mechanical bending test for flexible devices.",
        "feasibility": "No bending fixture or bend-cycle control exists in the current stack.",
        "defaults": {},
    },
    {
        "name": "Temperature-Dependent I-V",
        "category": "Mechanism Characterization",
        "status": STATUS_PARTIAL,
        "mode": MODE_SWEEP,
        "generator": "dciv_bipolar",
        "protocol_type": "DCIV",
        "description": "Repeat the same I-V list at externally controlled temperatures.",
        "feasibility": "Electrical acquisition is feasible; Arrhenius fitting is external.",
        "defaults": {
            "forward_voltage": 1.0,
            "reset_voltage": -1.0,
            "step_voltage": 0.02,
            "timer_delay": 0.001,
            "cycles": 1,
            "forming_cycle": False,
            "forming_voltage": 2.0,
            "peak_hold_steps": 0,
            "return_to_zero": True,
            "pos_compl": 0.001,
            "neg_compl": 0.01,
            "use_4way_split": True,
        },
    },
    {
        "name": "Conduction Mechanism Fitting",
        "category": "Mechanism Characterization",
        "status": STATUS_PARTIAL,
        "mode": MODE_INFO,
        "generator": None,
        "protocol_type": None,
        "description": "Fit I-V data to Ohmic, SCLC, Schottky, or Poole-Frenkel models.",
        "feasibility": "Data collection is supported; fitting is not built into this app.",
        "defaults": {},
    },
    {
        "name": "Impedance / EIS",
        "category": "Mechanism Characterization",
        "status": STATUS_NOT_READY,
        "mode": MODE_INFO,
        "generator": None,
        "protocol_type": None,
        "description": "Frequency-domain impedance spectroscopy.",
        "feasibility": "No impedance-analysis instrument control exists in the current setup.",
        "defaults": {},
    },
    {
        "name": "Potentiation / Depression (LTP/LTD)",
        "category": "Neuromorphic",
        "status": STATUS_READY,
        "mode": MODE_PULSE,
        "generator": "ltp_ltd",
        "protocol_type": "PULSE",
        "description": "Repeated potentiation followed by repeated depression pulses.",
        "feasibility": "Supported as a fixed-width pulse-slot sequence.",
        "defaults": {
            "base_width": 0.001,
            "write_voltage": 0.9,
            "write_width": 0.001,
            "write_gap": 0.001,
            "write_pulses": 25,
            "read_voltage": 0.1,
            "read_width": 0.001,
            "read_gap": 0.001,
            "read_pulses": 1,
            "erase_voltage": -0.9,
            "erase_width": 0.001,
            "erase_gap": 0.001,
            "erase_pulses": 25,
            "pulse_cycles": 1,
            "cycle_gap": 0.0,
            "initial_gap": 0.0,
            "final_read_block": False,
            "pulse_compliance": 0.005,
            "set_acquire_delay": 0.0006,
        },
    },
    {
        "name": "Paired-Pulse Facilitation (PPF)",
        "category": "Neuromorphic",
        "status": STATUS_PARTIAL,
        "mode": MODE_PULSE,
        "generator": "ppf",
        "protocol_type": "PULSE",
        "description": "Double-pulse sequence with configurable inter-pulse gap.",
        "feasibility": "Supported in quantized slots; metric extraction remains manual.",
        "defaults": {
            "base_width": 0.001,
            "write_voltage": 0.8,
            "write_width": 0.002,
            "write_gap": 0.01,
            "write_pulses": 2,
            "read_voltage": 0.1,
            "read_width": 0.001,
            "read_gap": 0.001,
            "read_pulses": 1,
            "erase_voltage": 0.0,
            "erase_width": 0.001,
            "erase_gap": 0.0,
            "erase_pulses": 0,
            "pulse_cycles": 20,
            "cycle_gap": 0.1,
            "initial_gap": 0.0,
            "final_read_block": False,
            "pulse_compliance": 0.002,
            "set_acquire_delay": 0.0006,
        },
    },
    {
        "name": "Spike-Timing-Dependent Plasticity (STDP)",
        "category": "Neuromorphic",
        "status": STATUS_NOT_READY,
        "mode": MODE_INFO,
        "generator": None,
        "protocol_type": None,
        "description": "True pre/post spike timing control.",
        "feasibility": "The current single-list protocol runner does not encode full STDP timing semantics.",
        "defaults": {},
    },
    {
        "name": "Probability-Voltage Sigmoid",
        "category": "p-bit / TRNG",
        "status": STATUS_READY,
        "mode": MODE_PULSE,
        "generator": "bias_stress",
        "protocol_type": "PULSE",
        "description": "Repeated biasing across a voltage range to collect a full probability-voltage sigmoid.",
        "feasibility": "Single-voltage preview and batch voltage-range export are both supported in this app.",
        "defaults": {
            "base_width": 0.001,
            "write_voltage": 0.5,
            "sigmoid_start_voltage": -0.5,
            "sigmoid_stop_voltage": 0.5,
            "sigmoid_step_voltage": 0.05,
            "write_width": 0.01,
            "write_gap": 0.005,
            "write_pulses": 1,
            "read_voltage": 0.1,
            "read_width": 0.001,
            "read_gap": 0.001,
            "read_pulses": 1,
            "erase_voltage": 0.0,
            "erase_width": 0.001,
            "erase_gap": 0.0,
            "erase_pulses": 0,
            "pulse_cycles": 200,
            "cycle_gap": 0.0,
            "initial_gap": 0.0,
            "final_read_block": False,
            "pulse_compliance": 0.002,
            "set_acquire_delay": 0.0006,
        },
    },
    {
        "name": "TRNG Randomness Test Campaign",
        "category": "p-bit / TRNG",
        "status": STATUS_NOT_READY,
        "mode": MODE_INFO,
        "generator": None,
        "protocol_type": None,
        "description": "NIST-style bitstream randomness validation.",
        "feasibility": "No bitstream pipeline or NIST test implementation is currently integrated.",
        "defaults": {},
    },
]

TEST_BY_NAME: Dict[str, Dict[str, object]] = {entry["name"]: entry for entry in TEST_CATALOG}


def parse_float(value: str, label: str, allow_blank: bool = False, default: float | None = None) -> float | None:
    text = str(value).strip()
    if text == "":
        if allow_blank:
            return default
        raise ValueError(f"{label} is required.")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number.") from exc


def parse_int(value: str, label: str, allow_blank: bool = False, default: int | None = None) -> int | None:
    text = str(value).strip()
    if text == "":
        if allow_blank:
            return default
        raise ValueError(f"{label} is required.")
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer.") from exc


def inclusive_ramp(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("Step voltage must be greater than zero.")
    direction = 1 if end >= start else -1
    actual_step = step * direction
    values: List[float] = []
    current = float(start)

    while True:
        values.append(round(current, 12))
        current += actual_step
        if direction > 0 and current > end + EPSILON:
            break
        if direction < 0 and current < end - EPSILON:
            break

    if abs(values[-1] - end) > EPSILON:
        values.append(round(float(end), 12))
    return values


def add_hold_points(sequence: List[float], hold_value: float, hold_steps: int) -> List[float]:
    if hold_steps <= 0:
        return sequence
    return sequence + [round(float(hold_value), 12)] * hold_steps


def build_bidirectional_cycle(
    forward_voltage: float,
    reset_voltage: float,
    step_voltage: float,
    peak_hold_steps: int = 0,
) -> List[float]:
    positive_up = inclusive_ramp(0.0, forward_voltage, step_voltage)
    positive_down = positive_up[-2::-1] if len(positive_up) > 1 else []
    positive_leg = add_hold_points(positive_up, forward_voltage, peak_hold_steps) + positive_down

    negative_down = inclusive_ramp(0.0, reset_voltage, step_voltage)
    negative_up = negative_down[-2::-1] if len(negative_down) > 1 else []
    negative_leg = add_hold_points(negative_down, reset_voltage, peak_hold_steps) + negative_up

    return positive_leg + negative_leg


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
) -> Tuple[List[float], List[float]]:
    voltages: List[float] = []
    times: List[float] = []
    current_time = 0.0

    if forming_cycle:
        if forming_voltage is None:
            raise ValueError("Forming voltage is required when forming is enabled.")
        forming_sequence = build_bidirectional_cycle(
            forming_voltage,
            reset_voltage,
            step_voltage,
            peak_hold_steps=peak_hold_steps,
        )
        for voltage in forming_sequence:
            voltages.append(voltage)
            times.append(round(current_time, 12))
            current_time += timer_delay

    for _ in range(cycles):
        cycle_sequence = build_bidirectional_cycle(
            forward_voltage,
            reset_voltage,
            step_voltage,
            peak_hold_steps=peak_hold_steps,
        )
        for voltage in cycle_sequence:
            voltages.append(voltage)
            times.append(round(current_time, 12))
            current_time += timer_delay

    if return_to_zero and voltages and abs(voltages[-1]) > EPSILON:
        voltages.append(0.0)
        times.append(round(current_time, 12))

    return times, voltages


def slots_from_seconds(duration_s: float, base_width_s: float) -> int:
    if duration_s < 0:
        raise ValueError("Durations cannot be negative.")
    if base_width_s <= 0:
        raise ValueError("Base pulse width must be greater than zero.")
    if duration_s == 0:
        return 0
    return max(1, int(round(duration_s / base_width_s)))


def append_slots(voltages: List[float], voltage: float, slot_count: int) -> None:
    if slot_count <= 0:
        return
    voltages.extend([round(float(voltage), 12)] * slot_count)


def build_times_for_slots(slot_count: int, base_width_s: float) -> List[float]:
    return [round(index * base_width_s, 12) for index in range(slot_count)]


def generate_endurance_slots(params: Dict[str, float | int | bool]) -> Tuple[List[float], List[float], List[str]]:
    base = float(params["base_width"])
    write_width = slots_from_seconds(float(params["write_width"]), base)
    write_gap = slots_from_seconds(float(params["write_gap"]), base)
    read_width = slots_from_seconds(float(params["read_width"]), base)
    read_gap = slots_from_seconds(float(params["read_gap"]), base)
    erase_width = slots_from_seconds(float(params["erase_width"]), base)
    erase_gap = slots_from_seconds(float(params["erase_gap"]), base)
    cycle_gap = slots_from_seconds(float(params["cycle_gap"]), base)
    initial_gap = slots_from_seconds(float(params["initial_gap"]), base)

    voltages: List[float] = []
    notes = [f"Pulse timings are quantized to {base:.6g} s slots for protocol compatibility."]
    append_slots(voltages, 0.0, initial_gap)

    for cycle_index in range(int(params["pulse_cycles"])):
        for _ in range(int(params["write_pulses"])):
            append_slots(voltages, float(params["write_voltage"]), write_width)
            append_slots(voltages, 0.0, write_gap)
        for _ in range(int(params["read_pulses"])):
            append_slots(voltages, float(params["read_voltage"]), read_width)
            append_slots(voltages, 0.0, read_gap)
        for _ in range(int(params["erase_pulses"])):
            append_slots(voltages, float(params["erase_voltage"]), erase_width)
            append_slots(voltages, 0.0, erase_gap)
        if bool(params["final_read_block"]):
            for _ in range(int(params["read_pulses"])):
                append_slots(voltages, float(params["read_voltage"]), read_width)
                append_slots(voltages, 0.0, read_gap)
        if cycle_index < int(params["pulse_cycles"]) - 1:
            append_slots(voltages, 0.0, cycle_gap)

    return build_times_for_slots(len(voltages), base), voltages, notes


def generate_bias_stress_slots(params: Dict[str, float | int | bool]) -> Tuple[List[float], List[float], List[str]]:
    base = float(params["base_width"])
    stress_width = slots_from_seconds(float(params["write_width"]), base)
    stress_gap = slots_from_seconds(float(params["write_gap"]), base)
    read_width = slots_from_seconds(float(params["read_width"]), base)
    read_gap = slots_from_seconds(float(params["read_gap"]), base)
    cycle_gap = slots_from_seconds(float(params["cycle_gap"]), base)
    initial_gap = slots_from_seconds(float(params["initial_gap"]), base)

    voltages: List[float] = []
    notes = [f"Bias-stress waveform is quantized to {base:.6g} s slots."]
    append_slots(voltages, 0.0, initial_gap)

    for cycle_index in range(int(params["pulse_cycles"])):
        append_slots(voltages, float(params["write_voltage"]), stress_width)
        append_slots(voltages, 0.0, stress_gap)
        for _ in range(int(params["read_pulses"])):
            append_slots(voltages, float(params["read_voltage"]), read_width)
            append_slots(voltages, 0.0, read_gap)
        if cycle_index < int(params["pulse_cycles"]) - 1:
            append_slots(voltages, 0.0, cycle_gap)

    return build_times_for_slots(len(voltages), base), voltages, notes


def generate_ltp_ltd_slots(params: Dict[str, float | int | bool]) -> Tuple[List[float], List[float], List[str]]:
    base = float(params["base_width"])
    write_width = slots_from_seconds(float(params["write_width"]), base)
    write_gap = slots_from_seconds(float(params["write_gap"]), base)
    read_width = slots_from_seconds(float(params["read_width"]), base)
    read_gap = slots_from_seconds(float(params["read_gap"]), base)
    erase_width = slots_from_seconds(float(params["erase_width"]), base)
    erase_gap = slots_from_seconds(float(params["erase_gap"]), base)

    voltages: List[float] = []
    notes = [f"LTP/LTD pulse train is quantized to {base:.6g} s slots."]

    for _ in range(int(params["write_pulses"])):
        append_slots(voltages, float(params["write_voltage"]), write_width)
        append_slots(voltages, 0.0, write_gap)
        for _ in range(int(params["read_pulses"])):
            append_slots(voltages, float(params["read_voltage"]), read_width)
            append_slots(voltages, 0.0, read_gap)

    for _ in range(int(params["erase_pulses"])):
        append_slots(voltages, float(params["erase_voltage"]), erase_width)
        append_slots(voltages, 0.0, erase_gap)
        for _ in range(int(params["read_pulses"])):
            append_slots(voltages, float(params["read_voltage"]), read_width)
            append_slots(voltages, 0.0, read_gap)

    return build_times_for_slots(len(voltages), base), voltages, notes


def generate_ppf_slots(params: Dict[str, float | int | bool]) -> Tuple[List[float], List[float], List[str]]:
    base = float(params["base_width"])
    spike_width = slots_from_seconds(float(params["write_width"]), base)
    gap_width = slots_from_seconds(float(params["write_gap"]), base)
    read_width = slots_from_seconds(float(params["read_width"]), base)
    read_gap = slots_from_seconds(float(params["read_gap"]), base)
    cycle_gap = slots_from_seconds(float(params["cycle_gap"]), base)

    voltages: List[float] = []
    notes = [f"PPF timing is represented with {base:.6g} s slots."]

    for cycle_index in range(int(params["pulse_cycles"])):
        append_slots(voltages, float(params["write_voltage"]), spike_width)
        append_slots(voltages, 0.0, gap_width)
        append_slots(voltages, float(params["write_voltage"]), spike_width)
        append_slots(voltages, 0.0, gap_width)
        for _ in range(int(params["read_pulses"])):
            append_slots(voltages, float(params["read_voltage"]), read_width)
            append_slots(voltages, 0.0, read_gap)
        if cycle_index < int(params["pulse_cycles"]) - 1:
            append_slots(voltages, 0.0, cycle_gap)

    return build_times_for_slots(len(voltages), base), voltages, notes


def generate_retention_read_slots(params: Dict[str, float | int | bool]) -> Tuple[List[float], List[float], List[str]]:
    base = float(params["base_width"])
    write_width = slots_from_seconds(float(params["write_width"]), base)
    write_gap = slots_from_seconds(float(params["write_gap"]), base)
    read_width = slots_from_seconds(float(params["read_width"]), base)
    read_gap = slots_from_seconds(float(params["read_gap"]), base)
    initial_gap = slots_from_seconds(float(params["initial_gap"]), base)

    voltages: List[float] = []
    notes = [
        f"Retention timing is quantized to {base:.6g} s slots.",
        "Long retention intervals can create very large voltage lists under the current pulse backend.",
    ]

    append_slots(voltages, 0.0, initial_gap)
    for _ in range(int(params["write_pulses"])):
        append_slots(voltages, float(params["write_voltage"]), write_width)
        append_slots(voltages, 0.0, write_gap)
    for _ in range(int(params["read_pulses"])):
        append_slots(voltages, float(params["read_voltage"]), read_width)
        append_slots(voltages, 0.0, read_gap)

    return build_times_for_slots(len(voltages), base), voltages, notes


def build_summary(
    test_name: str,
    mode: str,
    times: List[float],
    voltages: List[float],
    notes: List[str] | None = None,
) -> str:
    if not times or not voltages:
        return f"{test_name}\nNo data generated."

    duration = times[-1] if len(times) > 1 else 0.0
    positive_points = sum(1 for value in voltages if value > EPSILON)
    negative_points = sum(1 for value in voltages if value < -EPSILON)
    zero_points = len(voltages) - positive_points - negative_points

    lines = [
        test_name,
        f"Generator mode: {mode}",
        f"Points: {len(voltages)}",
        f"Duration: {duration:.6f} s",
        f"Voltage window: {min(voltages):.4f} V to {max(voltages):.4f} V",
        f"Positive / Negative / Zero points: {positive_points} / {negative_points} / {zero_points}",
    ]
    if notes:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines)


def plot_preview(times: List[float], voltages: List[float], title: str) -> None:
    plt.figure(figsize=(9, 4.5), dpi=110)
    plt.step(times, voltages, where="post")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()


def save_csv(times: List[float], voltages: List[float], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time (s)", "Voltage (V)"])
        for time_value, voltage_value in zip(times, voltages):
            writer.writerow([time_value, voltage_value])


def slugify(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in (" ", "-", "/"):
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug or "test"


def voltage_series(start: float, stop: float, step: float) -> List[float]:
    if abs(step) <= EPSILON:
        raise ValueError("Voltage step must be non-zero.")
    direction = 1 if stop >= start else -1
    actual_step = abs(step) * direction
    values: List[float] = []
    current = float(start)

    while True:
        values.append(round(current, 12))
        current += actual_step
        if direction > 0 and current > stop + EPSILON:
            break
        if direction < 0 and current < stop - EPSILON:
            break

    if abs(values[-1] - stop) > EPSILON:
        values.append(round(float(stop), 12))
    return values
class TestmakerApp:
    def __init__(self) -> None:
        if sg is None:
            raise ImportError("PySimpleGUI is required to launch Testmaker.")
        sg.theme("SystemDefault")
        self.generated: Dict[str, object] | None = None
        self.last_saved_csv_path: str = ""
        self.window = sg.Window(
            "ViPSA Testmaker",
            self._build_layout(),
            finalize=True,
            resizable=True,
            size=(1320, 860),
        )
        self._select_initial_test()

    def _build_layout(self):
        catalog_items = [f"{test['category']} :: {test['name']}" for test in TEST_CATALOG]

        left_column = [
            [sg.Text("Test Catalog", font=("Arial", 12, "bold"))],
            [sg.Listbox(catalog_items, key="-TEST_LIST-", size=(42, 32), enable_events=True, expand_y=True, expand_x=True)],
        ]

        details_frame = sg.Frame(
            "Selected Test",
            [
                [sg.Text("Name:", size=(8, 1)), sg.Text("", key="-TEST_NAME-", size=(70, 1), font=("Arial", 10, "bold"))],
                [sg.Text("Category:", size=(8, 1)), sg.Text("", key="-TEST_CATEGORY-", size=(70, 1))],
                [sg.Text("Status:", size=(8, 1)), sg.Text("", key="-TEST_STATUS-", size=(28, 1)), sg.Text("Mode:", size=(6, 1)), sg.Text("", key="-TEST_MODE-", size=(20, 1))],
                [sg.Multiline("", key="-TEST_DETAIL-", size=(88, 8), disabled=True, autoscroll=False, expand_x=True)],
            ],
            expand_x=True,
        )

        sweep_frame = sg.Frame(
            "Sweep Parameters",
            [
                [
                    sg.Text("Forward V", size=(14, 1)), sg.Input("1.5", key="SW_FORWARD_V", size=(10, 1)),
                    sg.Text("Reset V", size=(12, 1)), sg.Input("-1.5", key="SW_RESET_V", size=(10, 1)),
                    sg.Text("Step V", size=(10, 1)), sg.Input("0.05", key="SW_STEP_V", size=(10, 1)),
                    sg.Text("Delay (s)", size=(10, 1)), sg.Input("0.001", key="SW_DELAY", size=(10, 1)),
                ],
                [
                    sg.Text("Cycles", size=(14, 1)), sg.Input("2", key="SW_CYCLES", size=(10, 1)),
                    sg.Checkbox("Forming cycle", key="SW_FORMING", default=False, enable_events=True),
                    sg.Text("Forming V", size=(10, 1)), sg.Input("2.5", key="SW_FORMING_V", size=(10, 1), disabled=True),
                    sg.Text("Peak hold steps", size=(12, 1)), sg.Input("0", key="SW_HOLD", size=(10, 1)),
                    sg.Checkbox("Return to 0 V", key="SW_RETURN_ZERO", default=True),
                ],
                [
                    sg.Text("Pos compl (A)", size=(14, 1)), sg.Input("0.001", key="SW_POS_COMPL", size=(10, 1)),
                    sg.Text("Neg compl (A)", size=(12, 1)), sg.Input("0.01", key="SW_NEG_COMPL", size=(10, 1)),
                    sg.Checkbox("Use 4-way split", key="SW_4WAY", default=True),
                    sg.Text("SMU", size=(6, 1)),
                    sg.Combo(["Keithley2450", "KeysightB2901BL"], default_value="Keithley2450", key="SW_SMU", readonly=True, size=(16, 1)),
                ],
            ],
            expand_x=True,
        )

        pulse_frame = sg.Frame(
            "Pulse Parameters",
            [
                [
                    sg.Text("Base width (s)", size=(14, 1)), sg.Input("0.001", key="PU_BASE", size=(10, 1)),
                    sg.Text("Cycles", size=(10, 1)), sg.Input("100", key="PU_CYCLES", size=(10, 1)),
                    sg.Text("Cycle gap (s)", size=(12, 1)), sg.Input("0", key="PU_CYCLE_GAP", size=(10, 1)),
                    sg.Text("Initial gap (s)", size=(12, 1)), sg.Input("0", key="PU_INITIAL_GAP", size=(10, 1)),
                    sg.Checkbox("Final read block", key="PU_FINAL_READ", default=True),
                ],
                [
                    sg.Text("Write n", size=(14, 1)), sg.Input("1", key="PU_WRITE_N", size=(6, 1)),
                    sg.Text("Write V", size=(10, 1)), sg.Input("1.5", key="PU_WRITE_V", size=(8, 1)),
                    sg.Text("Write width (s)", size=(12, 1)), sg.Input("0.001", key="PU_WRITE_W", size=(10, 1)),
                    sg.Text("Write gap (s)", size=(12, 1)), sg.Input("0.001", key="PU_WRITE_G", size=(10, 1)),
                ],
                [
                    sg.Text("Read n", size=(14, 1)), sg.Input("1", key="PU_READ_N", size=(6, 1)),
                    sg.Text("Read V", size=(10, 1)), sg.Input("0.1", key="PU_READ_V", size=(8, 1)),
                    sg.Text("Read width (s)", size=(12, 1)), sg.Input("0.001", key="PU_READ_W", size=(10, 1)),
                    sg.Text("Read gap (s)", size=(12, 1)), sg.Input("0.001", key="PU_READ_G", size=(10, 1)),
                ],
                [
                    sg.Text("Erase n", size=(14, 1)), sg.Input("1", key="PU_ERASE_N", size=(6, 1)),
                    sg.Text("Erase V", size=(10, 1)), sg.Input("-1.8", key="PU_ERASE_V", size=(8, 1)),
                    sg.Text("Erase width (s)", size=(12, 1)), sg.Input("0.001", key="PU_ERASE_W", size=(10, 1)),
                    sg.Text("Erase gap (s)", size=(12, 1)), sg.Input("0.001", key="PU_ERASE_G", size=(10, 1)),
                ],
                [
                    sg.Text("Compliance (A)", size=(14, 1)), sg.Input("0.01", key="PU_COMPL", size=(10, 1)),
                    sg.Text("Acquire delay (s)", size=(14, 1)), sg.Input("0.0006", key="PU_ACQ_DELAY", size=(10, 1)),
                    sg.Text("SMU", size=(6, 1)),
                    sg.Combo(["KeysightB2901BL", "Keithley2450"], default_value="KeysightB2901BL", key="PU_SMU", readonly=True, size=(16, 1)),
                ],
            ],
            expand_x=True,
        )

        sigmoid_frame = sg.Frame(
            "Voltage Sigmoid Collection",
            [
                [
                    sg.Text("Start V", size=(14, 1)), sg.Input("-0.5", key="SIG_START_V", size=(10, 1)),
                    sg.Text("Stop V", size=(10, 1)), sg.Input("0.5", key="SIG_STOP_V", size=(10, 1)),
                    sg.Text("Step V", size=(10, 1)), sg.Input("0.05", key="SIG_STEP_V", size=(10, 1)),
                    sg.Button("Export Sigmoid Collection", key="-EXPORT_SIGMOID-"),
                ],
                [sg.Text("For each voltage in the range, Testmaker will save one pulse CSV and one protocol JSON.", size=(90, 1))],
            ],
            expand_x=True,
        )

        actions_frame = sg.Frame(
            "Actions",
            [
                [sg.Button("Load Test Defaults", key="-LOAD_DEFAULTS-"), sg.Button("Preview", key="-PREVIEW-"), sg.Button("Save CSV", key="-SAVE_CSV-"), sg.Button("Export Protocol JSON", key="-EXPORT_PROTOCOL-")],
                [sg.Text("Last saved CSV:", size=(14, 1)), sg.Input("", key="-LAST_CSV-", size=(88, 1), readonly=True)],
                [sg.Checkbox("Prepend ALIGN step", key="-PROTO_ALIGN-", default=False), sg.Checkbox("Prepend APPROACH step", key="-PROTO_APPROACH-", default=False)],
                [sg.Multiline("", key="-SUMMARY-", size=(88, 10), disabled=True, autoscroll=False, expand_x=True)],
            ],
            expand_x=True,
        )

        right_column = [[details_frame], [sweep_frame], [pulse_frame], [sigmoid_frame], [actions_frame]]
        return [[sg.Column(left_column, expand_y=True), sg.VSeparator(), sg.Column(right_column, expand_x=True, expand_y=True)]]

    def _select_initial_test(self) -> None:
        self.window["-TEST_LIST-"].update(set_to_index=[0])
        first_label = f"{TEST_CATALOG[0]['category']} :: {TEST_CATALOG[0]['name']}"
        self._load_test_into_ui(first_label, load_defaults=True)

    def _get_selected_test(self) -> Dict[str, object]:
        selected = self.window["-TEST_LIST-"].get()
        if not selected:
            raise ValueError("Select a test first.")
        _, test_name = selected[0].split(" :: ", 1)
        return TEST_BY_NAME[test_name]

    def _load_test_into_ui(self, label: str, load_defaults: bool) -> None:
        _, test_name = label.split(" :: ", 1)
        test = TEST_BY_NAME[test_name]
        self.window["-TEST_NAME-"].update(test["name"])
        self.window["-TEST_CATEGORY-"].update(test["category"])
        self.window["-TEST_STATUS-"].update(test["status"])
        self.window["-TEST_MODE-"].update(test["mode"])
        self.window["-TEST_DETAIL-"].update(value=f"{test['description']}\n\nFeasibility:\n{test['feasibility']}")
        if load_defaults:
            self._apply_defaults(test.get("defaults", {}))

    def _apply_defaults(self, defaults: Dict[str, object]) -> None:
        if not defaults:
            return
        mapping = {
            "forward_voltage": "SW_FORWARD_V", "reset_voltage": "SW_RESET_V", "step_voltage": "SW_STEP_V",
            "timer_delay": "SW_DELAY", "cycles": "SW_CYCLES", "forming_cycle": "SW_FORMING",
            "forming_voltage": "SW_FORMING_V", "peak_hold_steps": "SW_HOLD", "return_to_zero": "SW_RETURN_ZERO",
            "pos_compl": "SW_POS_COMPL", "neg_compl": "SW_NEG_COMPL", "use_4way_split": "SW_4WAY",
            "base_width": "PU_BASE", "write_voltage": "PU_WRITE_V", "write_width": "PU_WRITE_W",
            "sigmoid_start_voltage": "SIG_START_V", "sigmoid_stop_voltage": "SIG_STOP_V", "sigmoid_step_voltage": "SIG_STEP_V",
            "write_gap": "PU_WRITE_G", "write_pulses": "PU_WRITE_N", "read_voltage": "PU_READ_V",
            "read_width": "PU_READ_W", "read_gap": "PU_READ_G", "read_pulses": "PU_READ_N",
            "erase_voltage": "PU_ERASE_V", "erase_width": "PU_ERASE_W", "erase_gap": "PU_ERASE_G",
            "erase_pulses": "PU_ERASE_N", "pulse_cycles": "PU_CYCLES", "cycle_gap": "PU_CYCLE_GAP",
            "initial_gap": "PU_INITIAL_GAP", "final_read_block": "PU_FINAL_READ", "pulse_compliance": "PU_COMPL",
            "set_acquire_delay": "PU_ACQ_DELAY",
        }
        for key, value in defaults.items():
            target = mapping.get(key)
            if target is None:
                continue
            if isinstance(value, bool):
                self.window[target].update(value=value)
            else:
                self.window[target].update(value=str(value))
        self.window["SW_FORMING_V"].update(disabled=not bool(self.window["SW_FORMING"].get()))

    def _get_sweep_params(self) -> Dict[str, object]:
        forward_voltage = parse_float(self.window["SW_FORWARD_V"].get(), "Forward voltage")
        reset_voltage = parse_float(self.window["SW_RESET_V"].get(), "Reset voltage")
        step_voltage = parse_float(self.window["SW_STEP_V"].get(), "Step voltage")
        timer_delay = parse_float(self.window["SW_DELAY"].get(), "Sweep delay")
        cycles = parse_int(self.window["SW_CYCLES"].get(), "Cycles")
        peak_hold_steps = parse_int(self.window["SW_HOLD"].get(), "Peak hold steps")
        forming_cycle = bool(self.window["SW_FORMING"].get())
        forming_voltage = parse_float(self.window["SW_FORMING_V"].get(), "Forming voltage", allow_blank=True, default=None)

        if forward_voltage is None or forward_voltage <= 0:
            raise ValueError("Forward voltage must be greater than zero.")
        if reset_voltage is None or reset_voltage >= 0:
            raise ValueError("Reset voltage must be negative.")
        if step_voltage is None or step_voltage <= 0:
            raise ValueError("Step voltage must be greater than zero.")
        if timer_delay is None or timer_delay <= 0:
            raise ValueError("Sweep delay must be greater than zero.")
        if cycles is None or cycles < 1:
            raise ValueError("Cycles must be at least 1.")
        if peak_hold_steps is None or peak_hold_steps < 0:
            raise ValueError("Peak hold steps cannot be negative.")
        if forming_cycle:
            if forming_voltage is None or forming_voltage <= 0:
                raise ValueError("Forming voltage must be greater than zero when forming is enabled.")
            if forming_voltage < forward_voltage:
                raise ValueError("Forming voltage should be at least as large as forward voltage.")

        return {
            "forward_voltage": forward_voltage,
            "reset_voltage": reset_voltage,
            "step_voltage": step_voltage,
            "timer_delay": timer_delay,
            "forming_cycle": forming_cycle,
            "forming_voltage": forming_voltage,
            "cycles": cycles,
            "peak_hold_steps": peak_hold_steps,
            "return_to_zero": bool(self.window["SW_RETURN_ZERO"].get()),
            "pos_compl": parse_float(self.window["SW_POS_COMPL"].get(), "Positive compliance"),
            "neg_compl": parse_float(self.window["SW_NEG_COMPL"].get(), "Negative compliance"),
            "use_4way_split": bool(self.window["SW_4WAY"].get()),
            "smu_select": self.window["SW_SMU"].get(),
        }

    def _get_pulse_params(self) -> Dict[str, object]:
        params = {
            "base_width": parse_float(self.window["PU_BASE"].get(), "Base width"),
            "write_pulses": parse_int(self.window["PU_WRITE_N"].get(), "Write pulses"),
            "write_voltage": parse_float(self.window["PU_WRITE_V"].get(), "Write voltage"),
            "write_width": parse_float(self.window["PU_WRITE_W"].get(), "Write width"),
            "write_gap": parse_float(self.window["PU_WRITE_G"].get(), "Write gap"),
            "read_pulses": parse_int(self.window["PU_READ_N"].get(), "Read pulses"),
            "read_voltage": parse_float(self.window["PU_READ_V"].get(), "Read voltage"),
            "read_width": parse_float(self.window["PU_READ_W"].get(), "Read width"),
            "read_gap": parse_float(self.window["PU_READ_G"].get(), "Read gap"),
            "erase_pulses": parse_int(self.window["PU_ERASE_N"].get(), "Erase pulses"),
            "erase_voltage": parse_float(self.window["PU_ERASE_V"].get(), "Erase voltage"),
            "erase_width": parse_float(self.window["PU_ERASE_W"].get(), "Erase width"),
            "erase_gap": parse_float(self.window["PU_ERASE_G"].get(), "Erase gap"),
            "pulse_cycles": parse_int(self.window["PU_CYCLES"].get(), "Pulse cycles"),
            "cycle_gap": parse_float(self.window["PU_CYCLE_GAP"].get(), "Cycle gap"),
            "initial_gap": parse_float(self.window["PU_INITIAL_GAP"].get(), "Initial gap"),
            "final_read_block": bool(self.window["PU_FINAL_READ"].get()),
            "pulse_compliance": parse_float(self.window["PU_COMPL"].get(), "Pulse compliance"),
            "set_acquire_delay": parse_float(self.window["PU_ACQ_DELAY"].get(), "Acquire delay"),
            "smu_select": self.window["PU_SMU"].get(),
        }

        for key in ("base_width", "write_width", "write_gap", "read_width", "read_gap", "erase_width", "erase_gap", "cycle_gap", "initial_gap", "set_acquire_delay"):
            value = params[key]
            if value is None or value < 0:
                raise ValueError(f"{key.replace('_', ' ').title()} must be non-negative.")
        if params["base_width"] == 0:
            raise ValueError("Base width must be greater than zero.")
        if params["set_acquire_delay"] >= params["base_width"]:
            raise ValueError("Acquire delay must be smaller than the base width.")

        for key in ("write_pulses", "read_pulses", "erase_pulses", "pulse_cycles"):
            value = params[key]
            if value is None or value < 0:
                raise ValueError(f"{key.replace('_', ' ').title()} cannot be negative.")
        if params["pulse_cycles"] < 1:
            raise ValueError("Pulse cycles must be at least 1.")
        return params

    def _get_sigmoid_range_params(self) -> Dict[str, float]:
        start_voltage = parse_float(self.window["SIG_START_V"].get(), "Sigmoid start voltage")
        stop_voltage = parse_float(self.window["SIG_STOP_V"].get(), "Sigmoid stop voltage")
        step_voltage = parse_float(self.window["SIG_STEP_V"].get(), "Sigmoid step voltage")
        if start_voltage is None or stop_voltage is None or step_voltage is None:
            raise ValueError("All sigmoid range fields are required.")
        if abs(step_voltage) <= EPSILON:
            raise ValueError("Sigmoid step voltage must be non-zero.")
        return {
            "sigmoid_start_voltage": start_voltage,
            "sigmoid_stop_voltage": stop_voltage,
            "sigmoid_step_voltage": abs(step_voltage),
        }

    def _generate_for_selected_test(self) -> Dict[str, object]:
        test = self._get_selected_test()
        mode = str(test["mode"])
        generator = test.get("generator")
        protocol_type = test.get("protocol_type")
        if mode == MODE_INFO or not generator:
            raise ValueError("This test does not have a direct list generator in the current app.")

        if mode == MODE_SWEEP:
            params = self._get_sweep_params()
            times, voltages = generate_voltage_data(
                forward_voltage=float(params["forward_voltage"]),
                reset_voltage=float(params["reset_voltage"]),
                step_voltage=float(params["step_voltage"]),
                timer_delay=float(params["timer_delay"]),
                forming_cycle=bool(params["forming_cycle"]),
                forming_voltage=float(params["forming_voltage"]) if params["forming_voltage"] is not None else None,
                cycles=int(params["cycles"]),
                peak_hold_steps=int(params["peak_hold_steps"]),
                return_to_zero=bool(params["return_to_zero"]),
            )
            notes = ["Current protocol compatibility is direct: exported step uses this CSV as sweep_path."]
            return {"test": test, "mode": mode, "protocol_type": protocol_type, "times": times, "voltages": voltages, "notes": notes, "params": params}

        params = self._get_pulse_params()
        if generator == "endurance":
            times, voltages, notes = generate_endurance_slots(params)
        elif generator == "bias_stress":
            times, voltages, notes = generate_bias_stress_slots(params)
        elif generator == "ltp_ltd":
            times, voltages, notes = generate_ltp_ltd_slots(params)
        elif generator == "ppf":
            times, voltages, notes = generate_ppf_slots(params)
        elif generator == "retention_read":
            times, voltages, notes = generate_retention_read_slots(params)
        else:
            raise ValueError(f"Unsupported generator '{generator}'.")
        notes.append("Exported pulse protocol uses pulse_width = base_width and fixed slot timing.")
        return {"test": test, "mode": mode, "protocol_type": protocol_type, "times": times, "voltages": voltages, "notes": notes, "params": params}

    def _update_summary(self, generated: Dict[str, object]) -> None:
        summary = build_summary(
            test_name=str(generated["test"]["name"]),
            mode=str(generated["mode"]),
            times=list(generated["times"]),
            voltages=list(generated["voltages"]),
            notes=list(generated["notes"]),
        )
        self.window["-SUMMARY-"].update(value=summary)

    def _save_generated_csv(self) -> None:
        if not self.generated:
            raise ValueError("Generate or preview a test list first.")
        default_name = f"{slugify(str(self.generated['test']['name']))}.csv"
        filepath = sg.popup_get_file("Save CSV", save_as=True, no_window=True, default_extension=".csv", default_path=default_name, file_types=(("CSV Files", "*.csv"),))
        if not filepath:
            return
        save_csv(list(self.generated["times"]), list(self.generated["voltages"]), filepath)
        self.last_saved_csv_path = filepath
        self.window["-LAST_CSV-"].update(value=filepath)
        sg.popup_no_wait(f"Saved CSV to:\n{filepath}")

    def _build_protocol_steps_for_path(self, csv_path: str, generated: Dict[str, object]) -> List[Dict[str, object]]:
        protocol_steps: List[Dict[str, object]] = []
        if self.window["-PROTO_ALIGN-"].get():
            protocol_steps.append({"type": "ALIGN", "params": {"move": True, "zaber_corr": True, "recheck": True}})
        if self.window["-PROTO_APPROACH-"].get():
            protocol_steps.append({"type": "APPROACH", "params": {"step_size": 0.5, "test_voltage": 0.1, "lower_threshold": 1e-11, "upper_threshold": 5e-11, "max_attempts": 50, "delay": 1}})

        if generated["protocol_type"] == "DCIV":
            params = dict(generated["params"])
            protocol_steps.append({"type": "DCIV", "params": {"sweep_path": csv_path, "pos_compl": float(params["pos_compl"]), "neg_compl": float(params["neg_compl"]), "sweep_delay": float(params["timer_delay"]), "align": False, "approach": False, "smu_select": params["smu_select"], "use_4way_split": bool(params["use_4way_split"])}})
        elif generated["protocol_type"] == "PULSE":
            params = dict(generated["params"])
            protocol_steps.append({"type": "PULSE", "params": {"pulse_path": csv_path, "compliance": float(params["pulse_compliance"]), "pulse_width": float(params["base_width"]), "align": False, "approach": False, "smu_select": params["smu_select"], "set_acquire_delay": float(params["set_acquire_delay"])}})
        else:
            raise ValueError("This generated test is not protocol-exportable.")
        return protocol_steps

    def _export_sigmoid_collection(self) -> None:
        test = self._get_selected_test()
        if str(test["name"]) != "Probability-Voltage Sigmoid":
            raise ValueError("Sigmoid collection export is only for the Probability-Voltage Sigmoid test.")

        pulse_params = self._get_pulse_params()
        sigmoid_params = self._get_sigmoid_range_params()
        voltages = voltage_series(
            sigmoid_params["sigmoid_start_voltage"],
            sigmoid_params["sigmoid_stop_voltage"],
            sigmoid_params["sigmoid_step_voltage"],
        )

        target_dir = sg.popup_get_folder("Select output folder for sigmoid collection", no_window=True)
        if not target_dir:
            return
        os.makedirs(target_dir, exist_ok=True)

        manifest: Dict[str, object] = {
            "test": test["name"],
            "mode": "voltage_sigmoid_collection",
            "voltages": [],
            "count": len(voltages),
            "base_width_s": pulse_params["base_width"],
            "pulse_cycles": pulse_params["pulse_cycles"],
            "files": [],
        }

        for bias_voltage in voltages:
            run_params = dict(pulse_params)
            run_params["write_voltage"] = bias_voltage
            times, vlist, notes = generate_bias_stress_slots(run_params)
            notes.append("Generated as part of a voltage-sigmoid collection.")
            generated = {
                "test": test,
                "mode": MODE_PULSE,
                "protocol_type": "PULSE",
                "times": times,
                "voltages": vlist,
                "notes": notes,
                "params": run_params,
            }

            stem = f"{slugify(str(test['name']))}_v_{bias_voltage:+0.4f}".replace(".", "p").replace("+", "pos").replace("-", "neg")
            csv_path = os.path.join(target_dir, f"{stem}.csv")
            json_path = os.path.join(target_dir, f"{stem}_protocol.json")

            save_csv(times, vlist, csv_path)
            protocol_steps = self._build_protocol_steps_for_path(csv_path, generated)
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(protocol_steps, handle, indent=2)

            manifest["voltages"].append(bias_voltage)
            manifest["files"].append({"voltage": bias_voltage, "csv": csv_path, "protocol": json_path})

        manifest_path = os.path.join(target_dir, "sigmoid_collection_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        self.window["-SUMMARY-"].update(
            value="\n".join(
                [
                    "Probability-Voltage Sigmoid collection exported.",
                    f"Folder: {target_dir}",
                    f"Voltage points: {len(voltages)}",
                    f"Range: {voltages[0]:.4f} V to {voltages[-1]:.4f} V",
                    f"Manifest: {manifest_path}",
                ]
            )
        )
        sg.popup_no_wait(f"Saved sigmoid collection to:\n{target_dir}")

    def _build_protocol_steps(self) -> List[Dict[str, object]]:
        if not self.generated:
            raise ValueError("Generate and save a test list before exporting protocol JSON.")
        if not self.last_saved_csv_path:
            raise ValueError("Save the generated CSV first so the protocol can reference it.")
        return self._build_protocol_steps_for_path(self.last_saved_csv_path, self.generated)

    def _export_protocol_json(self) -> None:
        protocol_steps = self._build_protocol_steps()
        default_name = f"{slugify(str(self.generated['test']['name']))}_protocol.json"
        filepath = sg.popup_get_file("Save Protocol JSON", save_as=True, no_window=True, default_extension=".json", default_path=default_name, file_types=(("JSON Files", "*.json"),))
        if not filepath:
            return
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(protocol_steps, handle, indent=2)
        sg.popup_no_wait(f"Saved protocol JSON to:\n{filepath}")

    def run(self) -> None:
        while True:
            event, _ = self.window.read()
            if event == sg.WIN_CLOSED:
                break
            try:
                if event == "-TEST_LIST-":
                    selected = self.window["-TEST_LIST-"].get()
                    if selected:
                        self._load_test_into_ui(selected[0], load_defaults=False)
                elif event == "-LOAD_DEFAULTS-":
                    self._apply_defaults(dict(self._get_selected_test().get("defaults", {})))
                elif event == "SW_FORMING":
                    self.window["SW_FORMING_V"].update(disabled=not bool(self.window["SW_FORMING"].get()))
                elif event == "-PREVIEW-":
                    self.generated = self._generate_for_selected_test()
                    self._update_summary(self.generated)
                    plot_preview(list(self.generated["times"]), list(self.generated["voltages"]), str(self.generated["test"]["name"]))
                elif event == "-SAVE_CSV-":
                    if self.generated is None:
                        self.generated = self._generate_for_selected_test()
                        self._update_summary(self.generated)
                    self._save_generated_csv()
                elif event == "-EXPORT_SIGMOID-":
                    self._export_sigmoid_collection()
                elif event == "-EXPORT_PROTOCOL-":
                    self._export_protocol_json()
            except Exception as exc:
                sg.popup_error(str(exc))
        self.window.close()


def main() -> None:
    TestmakerApp().run()


if __name__ == "__main__":
    main()
