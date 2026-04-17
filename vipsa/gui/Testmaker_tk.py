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
import sys
import tkinter as tk
from typing import Dict, List, Tuple
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

try:
    from vipsa.analysis.sweep_generation import (
        SWEEP_MODE_OPTIONS,
        generate_voltage_data as generate_sweep_voltage_data,
    )
except ModuleNotFoundError:
    from sweep_generation import (
        SWEEP_MODE_OPTIONS,
        generate_voltage_data as generate_sweep_voltage_data,
    )


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
    sweep_mode: str = "positive_first",
) -> Tuple[List[float], List[float]]:
    times, voltages, _cycle_numbers = generate_sweep_voltage_data(
        forward_voltage=forward_voltage,
        reset_voltage=reset_voltage,
        step_voltage=step_voltage,
        timer_delay=timer_delay,
        forming_cycle=forming_cycle,
        forming_voltage=forming_voltage,
        cycles=cycles,
        peak_hold_steps=peak_hold_steps,
        return_to_zero=return_to_zero,
        sweep_mode=sweep_mode,
    )
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
    def __init__(self, parent: tk.Misc | None = None, protocol_callback=None, close_callback=None) -> None:
        self.generated: Dict[str, object] | None = None
        self.last_saved_csv_path: str = ""
        self.protocol_callback = protocol_callback
        self.close_callback = close_callback
        self._standalone = parent is None
        self.root = tk.Tk() if self._standalone else tk.Toplevel(parent)
        self.root.title("ViPSA Testmaker")
        self.root.geometry("1320x860")
        self.root.minsize(1100, 760)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        if not self._standalone:
            self.root.transient(parent)
        self.vars: Dict[str, tk.Variable] = {}
        self.widgets: Dict[str, tk.Widget] = {}
        self.catalog_labels = [f"{test['category']} :: {test['name']}" for test in TEST_CATALOG]
        self._configure_styles()
        self._build_layout()
        self._select_initial_test()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=0)
        container.columnconfigure(2, weight=1)
        container.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(container, padding=(0, 0, 10, 0))
        left_panel.grid(row=0, column=0, sticky="ns")
        left_panel.rowconfigure(1, weight=1)

        ttk.Label(left_panel, text="Test Catalog", font=("Arial", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        catalog_frame = ttk.Frame(left_panel)
        catalog_frame.grid(row=1, column=0, sticky="nsew")
        catalog_frame.rowconfigure(0, weight=1)
        catalog_frame.columnconfigure(0, weight=1)

        test_list = tk.Listbox(catalog_frame, width=42, exportselection=False)
        test_list.grid(row=0, column=0, sticky="nsew")
        test_scroll = ttk.Scrollbar(catalog_frame, orient="vertical", command=test_list.yview)
        test_scroll.grid(row=0, column=1, sticky="ns")
        test_list.configure(yscrollcommand=test_scroll.set)
        for item in self.catalog_labels:
            test_list.insert(tk.END, item)
        test_list.bind("<<ListboxSelect>>", self._on_test_selected)
        self.widgets["-TEST_LIST-"] = test_list

        ttk.Separator(container, orient="vertical").grid(row=0, column=1, sticky="ns")

        right_panel = ttk.Frame(container, padding=(12, 0, 0, 0))
        right_panel.grid(row=0, column=2, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        for row_index in range(5):
            right_panel.rowconfigure(row_index, weight=0)
        right_panel.rowconfigure(4, weight=1)

        self._build_details_frame(right_panel).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._build_sweep_frame(right_panel).grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self._build_pulse_frame(right_panel).grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self._build_sigmoid_frame(right_panel).grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self._build_actions_frame(right_panel).grid(row=4, column=0, sticky="nsew")

    def _build_details_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Selected Test", padding=10)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

        ttk.Label(frame, text="Name:", width=10).grid(row=0, column=0, sticky="w")
        self.widgets["-TEST_NAME-"] = ttk.Label(frame, text="", font=("Arial", 10, "bold"))
        self.widgets["-TEST_NAME-"].grid(row=0, column=1, columnspan=3, sticky="w")

        ttk.Label(frame, text="Category:", width=10).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.widgets["-TEST_CATEGORY-"] = ttk.Label(frame, text="")
        self.widgets["-TEST_CATEGORY-"].grid(row=1, column=1, columnspan=3, sticky="w", pady=(4, 0))

        ttk.Label(frame, text="Status:", width=10).grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.widgets["-TEST_STATUS-"] = ttk.Label(frame, text="")
        self.widgets["-TEST_STATUS-"].grid(row=2, column=1, sticky="w", pady=(4, 0))
        ttk.Label(frame, text="Mode:", width=10).grid(row=2, column=2, sticky="w", pady=(4, 0), padx=(12, 0))
        self.widgets["-TEST_MODE-"] = ttk.Label(frame, text="")
        self.widgets["-TEST_MODE-"].grid(row=2, column=3, sticky="w", pady=(4, 0))

        detail_text = tk.Text(frame, height=8, wrap="word", state="disabled")
        detail_text.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.widgets["-TEST_DETAIL-"] = detail_text
        return frame

    def _build_sweep_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Sweep Parameters", padding=10)
        for column in range(8):
            frame.columnconfigure(column, weight=1 if column % 2 else 0)

        self._entry(frame, 0, 0, "Forward V", "SW_FORWARD_V", "1.5")
        self._entry(frame, 0, 2, "Reset V", "SW_RESET_V", "-1.5")
        self._entry(frame, 0, 4, "Step V", "SW_STEP_V", "0.05")
        self._entry(frame, 0, 6, "Delay (s)", "SW_DELAY", "0.001")

        self._entry(frame, 1, 0, "Cycles", "SW_CYCLES", "2")
        self._check(frame, 1, 2, "Forming cycle", "SW_FORMING", False, self._update_forming_enabled)
        self._entry(frame, 1, 4, "Forming V", "SW_FORMING_V", "2.5")
        self._entry(frame, 1, 6, "Peak hold steps", "SW_HOLD", "0")

        return_check = ttk.Checkbutton(
            frame,
            text="Return to 0 V",
            variable=self._bool_var("SW_RETURN_ZERO", True),
        )
        return_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(frame, text="Sweep mode").grid(row=2, column=2, sticky="w", pady=(6, 0))
        sweep_mode = ttk.Combobox(
            frame,
            textvariable=self._string_var("SW_MODE", "positive_first"),
            values=list(SWEEP_MODE_OPTIONS),
            state="readonly",
            width=22,
        )
        sweep_mode.grid(row=2, column=3, sticky="ew", pady=(6, 0))
        self.widgets["SW_MODE"] = sweep_mode
        ttk.Label(
            frame,
            text="Default keeps the current 0 -> +V -> 0 -> -V -> 0 pattern.",
        ).grid(row=2, column=4, columnspan=4, sticky="w", padx=(12, 0), pady=(6, 0))

        self._entry(frame, 3, 0, "Pos compl (A)", "SW_POS_COMPL", "0.001")
        self._entry(frame, 3, 2, "Neg compl (A)", "SW_NEG_COMPL", "0.01")
        self._check(frame, 3, 4, "Use 4-way split", "SW_4WAY", True, None)

        ttk.Label(frame, text="SMU").grid(row=3, column=6, sticky="w", pady=(6, 0))
        sweep_smu = ttk.Combobox(
            frame,
            textvariable=self._string_var("SW_SMU", "Keithley2450"),
            values=("Keithley2450", "KeysightB2901BL"),
            state="readonly",
            width=16,
        )
        sweep_smu.grid(row=3, column=7, sticky="ew", pady=(6, 0))
        self.widgets["SW_SMU"] = sweep_smu
        return frame

    def _build_pulse_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Pulse Parameters", padding=10)
        for column in range(10):
            frame.columnconfigure(column, weight=1 if column % 2 else 0)

        self._entry(frame, 0, 0, "Base width (s)", "PU_BASE", "0.001")
        self._entry(frame, 0, 2, "Cycles", "PU_CYCLES", "100")
        self._entry(frame, 0, 4, "Cycle gap (s)", "PU_CYCLE_GAP", "0")
        self._entry(frame, 0, 6, "Initial gap (s)", "PU_INITIAL_GAP", "0")
        final_read = ttk.Checkbutton(
            frame,
            text="Final read block",
            variable=self._bool_var("PU_FINAL_READ", True),
        )
        final_read.grid(row=0, column=8, columnspan=2, sticky="w")

        self._entry(frame, 1, 0, "Write n", "PU_WRITE_N", "1", width=8)
        self._entry(frame, 1, 2, "Write V", "PU_WRITE_V", "1.5", width=10)
        self._entry(frame, 1, 4, "Write width (s)", "PU_WRITE_W", "0.001")
        self._entry(frame, 1, 6, "Write gap (s)", "PU_WRITE_G", "0.001")

        self._entry(frame, 2, 0, "Read n", "PU_READ_N", "1", width=8)
        self._entry(frame, 2, 2, "Read V", "PU_READ_V", "0.1", width=10)
        self._entry(frame, 2, 4, "Read width (s)", "PU_READ_W", "0.001")
        self._entry(frame, 2, 6, "Read gap (s)", "PU_READ_G", "0.001")

        self._entry(frame, 3, 0, "Erase n", "PU_ERASE_N", "1", width=8)
        self._entry(frame, 3, 2, "Erase V", "PU_ERASE_V", "-1.8", width=10)
        self._entry(frame, 3, 4, "Erase width (s)", "PU_ERASE_W", "0.001")
        self._entry(frame, 3, 6, "Erase gap (s)", "PU_ERASE_G", "0.001")

        self._entry(frame, 4, 0, "Compliance (A)", "PU_COMPL", "0.01")
        self._entry(frame, 4, 2, "Acquire delay (s)", "PU_ACQ_DELAY", "0.0006")
        ttk.Label(frame, text="SMU").grid(row=4, column=4, sticky="w", pady=(6, 0))
        pulse_smu = ttk.Combobox(
            frame,
            textvariable=self._string_var("PU_SMU", "KeysightB2901BL"),
            values=("KeysightB2901BL", "Keithley2450"),
            state="readonly",
            width=16,
        )
        pulse_smu.grid(row=4, column=5, sticky="ew", pady=(6, 0))
        self.widgets["PU_SMU"] = pulse_smu
        return frame

    def _build_sigmoid_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Voltage Sigmoid Collection", padding=10)
        for column in range(7):
            frame.columnconfigure(column, weight=1 if column in (1, 3, 5) else 0)

        self._entry(frame, 0, 0, "Start V", "SIG_START_V", "-0.5")
        self._entry(frame, 0, 2, "Stop V", "SIG_STOP_V", "0.5")
        self._entry(frame, 0, 4, "Step V", "SIG_STEP_V", "0.05")
        ttk.Button(
            frame,
            text="Export Sigmoid Collection",
            command=lambda: self._handle_action(self._export_sigmoid_collection),
        ).grid(row=0, column=6, sticky="e")
        ttk.Label(
            frame,
            text="For each voltage in the range, Testmaker will save one pulse CSV and one protocol JSON.",
        ).grid(row=1, column=0, columnspan=7, sticky="w", pady=(6, 0))
        return frame

    def _build_actions_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        frame.columnconfigure(0, weight=1)

        actions = ttk.Frame(frame)
        actions.grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="Load Test Defaults", command=lambda: self._handle_action(self._load_defaults_for_selected)).pack(side="left")
        ttk.Button(actions, text="Preview", command=lambda: self._handle_action(self._preview)).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Save CSV", command=lambda: self._handle_action(self._save_csv_action)).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Export Protocol JSON", command=lambda: self._handle_action(self._export_protocol_json)).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Send To Viewfinder", command=lambda: self._handle_action(self._send_protocol_to_viewfinder)).pack(side="left", padx=(6, 0))

        last_csv_row = ttk.Frame(frame)
        last_csv_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        last_csv_row.columnconfigure(1, weight=1)
        ttk.Label(last_csv_row, text="Last saved CSV:", width=14).grid(row=0, column=0, sticky="w")
        last_csv = ttk.Entry(last_csv_row, textvariable=self._string_var("-LAST_CSV-", ""), state="readonly")
        last_csv.grid(row=0, column=1, sticky="ew")
        self.widgets["-LAST_CSV-"] = last_csv

        proto_row = ttk.Frame(frame)
        proto_row.grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Checkbutton(proto_row, text="Prepend ALIGN step", variable=self._bool_var("-PROTO_ALIGN-", False)).pack(side="left")
        ttk.Checkbutton(proto_row, text="Prepend APPROACH step", variable=self._bool_var("-PROTO_APPROACH-", False)).pack(side="left", padx=(12, 0))

        summary = tk.Text(frame, height=10, wrap="word", state="disabled")
        summary.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        frame.rowconfigure(3, weight=1)
        self.widgets["-SUMMARY-"] = summary
        return frame

    def _string_var(self, key: str, default: str) -> tk.StringVar:
        var = self.vars.get(key)
        if isinstance(var, tk.StringVar):
            return var
        new_var = tk.StringVar(value=default)
        self.vars[key] = new_var
        return new_var

    def _bool_var(self, key: str, default: bool) -> tk.BooleanVar:
        var = self.vars.get(key)
        if isinstance(var, tk.BooleanVar):
            return var
        new_var = tk.BooleanVar(value=default)
        self.vars[key] = new_var
        return new_var

    def _entry(
        self,
        parent: ttk.Frame,
        row: int,
        column: int,
        label: str,
        key: str,
        default: str,
        width: int = 12,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", pady=(6, 0) if row else 0)
        entry = ttk.Entry(parent, textvariable=self._string_var(key, default), width=width)
        entry.grid(row=row, column=column + 1, sticky="ew", padx=(6, 12), pady=(6, 0) if row else 0)
        self.widgets[key] = entry
        return entry

    def _check(
        self,
        parent: ttk.Frame,
        row: int,
        column: int,
        label: str,
        key: str,
        default: bool,
        command,
    ) -> ttk.Checkbutton:
        check = ttk.Checkbutton(parent, text=label, variable=self._bool_var(key, default), command=command)
        check.grid(row=row, column=column, columnspan=2, sticky="w", pady=(6, 0) if row else 0)
        self.widgets[key] = check
        return check

    def _set_text_widget(self, key: str, value: str) -> None:
        widget = self.widgets[key]
        if not isinstance(widget, tk.Text):
            raise TypeError(f"Widget {key} is not a text widget.")
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _set_label_text(self, key: str, value: str) -> None:
        widget = self.widgets[key]
        if isinstance(widget, ttk.Label):
            widget.configure(text=value)

    def _update_forming_enabled(self) -> None:
        state = "normal" if bool(self._bool_var("SW_FORMING", False).get()) else "disabled"
        entry = self.widgets.get("SW_FORMING_V")
        if isinstance(entry, ttk.Entry):
            entry.configure(state=state)

    def _clear_generated_state(self) -> None:
        self.generated = None
        self.last_saved_csv_path = ""
        self._string_var("-LAST_CSV-", "").set("")

    def _select_initial_test(self) -> None:
        test_list = self.widgets["-TEST_LIST-"]
        if isinstance(test_list, tk.Listbox):
            test_list.selection_set(0)
            test_list.activate(0)
            test_list.see(0)
        first_label = self.catalog_labels[0]
        self._load_test_into_ui(first_label, load_defaults=True)

    def _get_selected_test(self) -> Dict[str, object]:
        test_list = self.widgets["-TEST_LIST-"]
        if not isinstance(test_list, tk.Listbox):
            raise RuntimeError("Test list widget is unavailable.")
        selection = test_list.curselection()
        if not selection:
            raise ValueError("Select a test first.")
        _, test_name = test_list.get(selection[0]).split(" :: ", 1)
        return TEST_BY_NAME[test_name]

    def _load_test_into_ui(self, label: str, load_defaults: bool) -> None:
        _, test_name = label.split(" :: ", 1)
        test = TEST_BY_NAME[test_name]
        self._clear_generated_state()
        self._set_label_text("-TEST_NAME-", str(test["name"]))
        self._set_label_text("-TEST_CATEGORY-", str(test["category"]))
        self._set_label_text("-TEST_STATUS-", str(test["status"]))
        self._set_label_text("-TEST_MODE-", str(test["mode"]))
        self._set_text_widget("-TEST_DETAIL-", f"{test['description']}\n\nFeasibility:\n{test['feasibility']}")
        self._set_text_widget("-SUMMARY-", "")
        if load_defaults:
            self._apply_defaults(test.get("defaults", {}))

    def _apply_defaults(self, defaults: Dict[str, object]) -> None:
        if not defaults:
            return
        mapping = {
            "forward_voltage": "SW_FORWARD_V", "reset_voltage": "SW_RESET_V", "step_voltage": "SW_STEP_V",
            "timer_delay": "SW_DELAY", "cycles": "SW_CYCLES", "forming_cycle": "SW_FORMING",
            "forming_voltage": "SW_FORMING_V", "peak_hold_steps": "SW_HOLD", "return_to_zero": "SW_RETURN_ZERO",
            "sweep_mode": "SW_MODE",
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
            variable = self.vars.get(target)
            if isinstance(variable, tk.BooleanVar):
                variable.set(bool(value))
            else:
                self._string_var(target, "").set(str(value))
        self._update_forming_enabled()

    def _get_sweep_params(self) -> Dict[str, object]:
        forward_voltage = parse_float(self._string_var("SW_FORWARD_V", "").get(), "Forward voltage")
        reset_voltage = parse_float(self._string_var("SW_RESET_V", "").get(), "Reset voltage")
        step_voltage = parse_float(self._string_var("SW_STEP_V", "").get(), "Step voltage")
        timer_delay = parse_float(self._string_var("SW_DELAY", "").get(), "Sweep delay")
        cycles = parse_int(self._string_var("SW_CYCLES", "").get(), "Cycles")
        peak_hold_steps = parse_int(self._string_var("SW_HOLD", "").get(), "Peak hold steps")
        forming_cycle = bool(self._bool_var("SW_FORMING", False).get())
        forming_voltage = parse_float(self._string_var("SW_FORMING_V", "").get(), "Forming voltage", allow_blank=True, default=None)

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
            "return_to_zero": bool(self._bool_var("SW_RETURN_ZERO", True).get()),
            "sweep_mode": self._string_var("SW_MODE", "positive_first").get() or "positive_first",
            "pos_compl": parse_float(self._string_var("SW_POS_COMPL", "").get(), "Positive compliance"),
            "neg_compl": parse_float(self._string_var("SW_NEG_COMPL", "").get(), "Negative compliance"),
            "use_4way_split": bool(self._bool_var("SW_4WAY", True).get()),
            "smu_select": self._string_var("SW_SMU", "Keithley2450").get(),
        }

    def _get_pulse_params(self) -> Dict[str, object]:
        params = {
            "base_width": parse_float(self._string_var("PU_BASE", "").get(), "Base width"),
            "write_pulses": parse_int(self._string_var("PU_WRITE_N", "").get(), "Write pulses"),
            "write_voltage": parse_float(self._string_var("PU_WRITE_V", "").get(), "Write voltage"),
            "write_width": parse_float(self._string_var("PU_WRITE_W", "").get(), "Write width"),
            "write_gap": parse_float(self._string_var("PU_WRITE_G", "").get(), "Write gap"),
            "read_pulses": parse_int(self._string_var("PU_READ_N", "").get(), "Read pulses"),
            "read_voltage": parse_float(self._string_var("PU_READ_V", "").get(), "Read voltage"),
            "read_width": parse_float(self._string_var("PU_READ_W", "").get(), "Read width"),
            "read_gap": parse_float(self._string_var("PU_READ_G", "").get(), "Read gap"),
            "erase_pulses": parse_int(self._string_var("PU_ERASE_N", "").get(), "Erase pulses"),
            "erase_voltage": parse_float(self._string_var("PU_ERASE_V", "").get(), "Erase voltage"),
            "erase_width": parse_float(self._string_var("PU_ERASE_W", "").get(), "Erase width"),
            "erase_gap": parse_float(self._string_var("PU_ERASE_G", "").get(), "Erase gap"),
            "pulse_cycles": parse_int(self._string_var("PU_CYCLES", "").get(), "Pulse cycles"),
            "cycle_gap": parse_float(self._string_var("PU_CYCLE_GAP", "").get(), "Cycle gap"),
            "initial_gap": parse_float(self._string_var("PU_INITIAL_GAP", "").get(), "Initial gap"),
            "final_read_block": bool(self._bool_var("PU_FINAL_READ", True).get()),
            "pulse_compliance": parse_float(self._string_var("PU_COMPL", "").get(), "Pulse compliance"),
            "set_acquire_delay": parse_float(self._string_var("PU_ACQ_DELAY", "").get(), "Acquire delay"),
            "smu_select": self._string_var("PU_SMU", "KeysightB2901BL").get(),
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
        start_voltage = parse_float(self._string_var("SIG_START_V", "").get(), "Sigmoid start voltage")
        stop_voltage = parse_float(self._string_var("SIG_STOP_V", "").get(), "Sigmoid stop voltage")
        step_voltage = parse_float(self._string_var("SIG_STEP_V", "").get(), "Sigmoid step voltage")
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
                sweep_mode=str(params.get("sweep_mode", "positive_first")),
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
        self._set_text_widget("-SUMMARY-", summary)

    def _save_generated_csv(self) -> None:
        if not self.generated:
            raise ValueError("Generate or preview a test list first.")
        default_name = f"{slugify(str(self.generated['test']['name']))}.csv"
        filepath = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=(("CSV Files", "*.csv"),),
        )
        if not filepath:
            return
        self._save_generated_csv_to_path(filepath)

    def _save_generated_csv_to_path(self, filepath: str) -> None:
        if not self.generated:
            raise ValueError("Generate or preview a test list first.")
        save_csv(list(self.generated["times"]), list(self.generated["voltages"]), filepath)
        self.last_saved_csv_path = filepath
        self._string_var("-LAST_CSV-", "").set(filepath)
        messagebox.showinfo("Testmaker", f"Saved CSV to:\n{filepath}", parent=self.root)

    def _prompt_csv_path(self) -> str:
        if not self.generated:
            raise ValueError("Generate or preview a test list first.")
        default_name = f"{slugify(str(self.generated['test']['name']))}.csv"
        filepath = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save CSV For Viewfinder Protocol",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=(("CSV Files", "*.csv"),),
        )
        return filepath or ""

    def _build_protocol_steps_for_path(self, csv_path: str, generated: Dict[str, object]) -> List[Dict[str, object]]:
        protocol_steps: List[Dict[str, object]] = []
        if self._bool_var("-PROTO_ALIGN-", False).get():
            protocol_steps.append({"type": "ALIGN", "params": {"move": True, "zaber_corr": True, "recheck": True}})
        if self._bool_var("-PROTO_APPROACH-", False).get():
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

        target_dir = filedialog.askdirectory(
            parent=self.root,
            title="Select output folder for sigmoid collection",
            mustexist=False,
        )
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

        self._set_text_widget(
            "-SUMMARY-",
            "\n".join(
                [
                    "Probability-Voltage Sigmoid collection exported.",
                    f"Folder: {target_dir}",
                    f"Voltage points: {len(voltages)}",
                    f"Range: {voltages[0]:.4f} V to {voltages[-1]:.4f} V",
                    f"Manifest: {manifest_path}",
                ]
            ),
        )
        messagebox.showinfo("Testmaker", f"Saved sigmoid collection to:\n{target_dir}", parent=self.root)

    def _build_protocol_steps(self) -> List[Dict[str, object]]:
        if not self.generated:
            raise ValueError("Generate and save a test list before exporting protocol JSON.")
        if not self.last_saved_csv_path:
            raise ValueError("Save the generated CSV first so the protocol can reference it.")
        return self._build_protocol_steps_for_path(self.last_saved_csv_path, self.generated)

    def _export_protocol_json(self) -> None:
        protocol_steps = self._build_protocol_steps()
        default_name = f"{slugify(str(self.generated['test']['name']))}_protocol.json"
        filepath = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Protocol JSON",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=(("JSON Files", "*.json"),),
        )
        if not filepath:
            return
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(protocol_steps, handle, indent=2)
        messagebox.showinfo("Testmaker", f"Saved protocol JSON to:\n{filepath}", parent=self.root)

    def _send_protocol_to_viewfinder(self) -> None:
        if self.protocol_callback is None:
            raise ValueError("Viewfinder link not available in standalone Testmaker.")
        if self.generated is None:
            self.generated = self._generate_for_selected_test()
            self._update_summary(self.generated)
        if not self.last_saved_csv_path or not os.path.isfile(self.last_saved_csv_path):
            filepath = self._prompt_csv_path()
            if not filepath:
                return
            self._save_generated_csv_to_path(filepath)
        protocol_steps = self._build_protocol_steps()
        self.protocol_callback(protocol_steps, self.last_saved_csv_path, self.generated)
        messagebox.showinfo("Testmaker", "Protocol sent to Viewfinder.", parent=self.root)

    def _load_defaults_for_selected(self) -> None:
        self._apply_defaults(dict(self._get_selected_test().get("defaults", {})))

    def _preview(self) -> None:
        self.generated = self._generate_for_selected_test()
        self._update_summary(self.generated)
        plot_preview(list(self.generated["times"]), list(self.generated["voltages"]), str(self.generated["test"]["name"]))

    def _save_csv_action(self) -> None:
        if self.generated is None:
            self.generated = self._generate_for_selected_test()
            self._update_summary(self.generated)
        self._save_generated_csv()

    def _on_test_selected(self, _event=None) -> None:
        test_list = self.widgets["-TEST_LIST-"]
        if not isinstance(test_list, tk.Listbox):
            return
        selection = test_list.curselection()
        if selection:
            self._load_test_into_ui(test_list.get(selection[0]), load_defaults=False)

    def _handle_action(self, callback) -> None:
        try:
            callback()
        except Exception as exc:
            messagebox.showerror("Testmaker", str(exc), parent=self.root)

    def _on_close(self) -> None:
        if callable(self.close_callback):
            try:
                self.close_callback()
            except Exception:
                pass
        self.root.destroy()

    def run(self) -> None:
        if self._standalone:
            self.root.mainloop()


def main() -> None:
    TestmakerApp().run()


if __name__ == "__main__":
    main()
