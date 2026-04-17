import argparse
import sys
from typing import Iterable

import numpy as np
import pyvisa

from Source_Measure_Unit import Keithley707B, KeithleySMU, KeysightSMU


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def normalize_snapshot(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum() or ch == ",")


def assert_channels_present(snapshot: str, expected_channels: Iterable[str], label: str) -> None:
    normalized = normalize_snapshot(snapshot)
    missing = [channel for channel in expected_channels if channel.upper() not in normalized]
    if missing:
        raise RuntimeError(
            f"{label} route check failed. Expected {missing} in matrix snapshot '{snapshot}'."
        )


def print_preview(label: str, data) -> None:
    arr = np.asarray(data)
    print(f"{label}: {len(arr)} point(s)")
    if len(arr) > 0:
        print(arr[: min(5, len(arr))])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tiny 707B switch validation: Keithley tiny DCIV, then Keysight tiny pulse."
    )
    parser.add_argument("--switch-index", type=int, default=0, help="VISA resource index for the 707B.")
    parser.add_argument("--switch-address", default=None, help="Explicit VISA address for the 707B.")
    parser.add_argument("--keithley-index", type=int, default=1, help="VISA resource index for the Keithley 2450.")
    parser.add_argument("--keithley-address", default=None, help="Explicit VISA address for the Keithley 2450.")
    parser.add_argument("--keysight-index", type=int, default=2, help="VISA resource index for the Keysight SMU.")
    parser.add_argument("--keysight-address", default=None, help="Explicit VISA address for the Keysight SMU.")
    parser.add_argument("--slot", type=int, default=1, help="707B slot number for the matrix card.")
    parser.add_argument(
        "--dciv-voltages",
        default="0.0,0.05,0.0,-0.05,0.0",
        help="Comma-separated voltage list for the Keithley tiny DCIV-style test.",
    )
    parser.add_argument("--dciv-compliance", type=float, default=1e-3, help="Current compliance for the Keithley test.")
    parser.add_argument("--dciv-delay", type=float, default=0.01, help="Delay between Keithley DCIV points in seconds.")
    parser.add_argument("--dciv-nplc", type=float, default=0.01, help="Keithley NPLC for the tiny DCIV test.")
    parser.add_argument(
        "--pulse-voltages",
        default="0.10,0.0,0.10,0.0",
        help="Comma-separated pulse list for the Keysight test.",
    )
    parser.add_argument("--pulse-compliance", type=float, default=1e-3, help="Current compliance for the Keysight pulse test.")
    parser.add_argument("--pulse-width", type=float, default=0.01, help="Pulse width in seconds for the Keysight test.")
    parser.add_argument(
        "--pulse-acquire-delay",
        type=float,
        default=0.005,
        help="Acquisition delay in seconds for the Keysight pulse test.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    rm = pyvisa.ResourceManager()
    resources = list(rm.list_resources())
    print("VISA resources seen by PyVISA:")
    for idx, resource in enumerate(resources):
        print(f"  [{idx}] {resource}")

    switch = Keithley707B(device_no=args.switch_index, address=args.switch_address, slot=args.slot)
    keithley = KeithleySMU(device_no=args.keithley_index, address=args.keithley_address)
    keysight = KeysightSMU(device_no=args.keysight_index, address=args.keysight_address)

    dciv_voltages = parse_float_list(args.dciv_voltages)
    pulse_voltages = parse_float_list(args.pulse_voltages)

    expected_keithley = switch.get_route_channels("keithley")
    expected_keysight = switch.get_route_channels("keysight")

    print(f"Expected Keithley route: {expected_keithley}")
    print(f"Expected Keysight route: {expected_keysight}")

    try:
        print("\nOpening all channels before starting.")
        switch.open_all()
        print("Closed-channel snapshot:", switch.get_closed_channels())

        print("\nRouting matrix to Keithley.")
        switch.connect_keithley_smu()
        keithley_snapshot = switch.get_closed_channels()
        print("Closed-channel snapshot:", keithley_snapshot)
        assert_channels_present(keithley_snapshot, expected_keithley, "Keithley")

        keithley_data = keithley.run_read_probe(
            dciv_voltages,
            compliance=args.dciv_compliance,
            delay=args.dciv_delay,
            nplc=args.dciv_nplc,
            label="Keithley tiny DCIV",
        )
        print_preview("Keithley tiny DCIV data", keithley_data)

        print("\nOpening all channels after Keithley test.")
        switch.open_all()
        print("Closed-channel snapshot:", switch.get_closed_channels())

        print("\nRouting matrix to Keysight.")
        switch.connect_keysight_smu()
        keysight_snapshot = switch.get_closed_channels()
        print("Closed-channel snapshot:", keysight_snapshot)
        assert_channels_present(keysight_snapshot, expected_keysight, "Keysight")

        pulse_data = keysight.pulsed_measurement(
            csv_path=None,
            current_compliance=args.pulse_compliance,
            set_width=args.pulse_width,
            bare_list=pulse_voltages,
            set_acquire_delay=args.pulse_acquire_delay,
        )
        if pulse_data is None:
            raise RuntimeError("Keysight pulse test returned no data.")
        print_preview("Keysight tiny pulse data", pulse_data)

        print("\nValidation sequence completed.")
        return 0

    finally:
        print("\nOpening all channels at the end.")
        try:
            switch.open_all()
            print("Final closed-channel snapshot:", switch.get_closed_channels())
        except Exception as exc:
            print(f"Warning: could not open all channels at shutdown: {exc}")


if __name__ == "__main__":
    sys.exit(main())
