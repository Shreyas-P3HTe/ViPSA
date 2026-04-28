from __future__ import annotations

import argparse
import time

import pyvisa


DEFAULT_ADDRESS = "USB0::0x0957::0x8C18::MY51142764::INSTR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Force Keysight front terminal and use explicit spot-measure queries."
    )
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--voltage", type=float, default=0.1)
    parser.add_argument("--compliance", type=float, default=1e-3)
    parser.add_argument("--current-range", type=float, default=1e-3)
    parser.add_argument("--settle", type=float, default=0.02)
    parser.add_argument("--auto-current-range", action="store_true")
    return parser.parse_args()


def write(dev, command: str) -> None:
    print(f">> {command}")
    dev.write(command)


def query(dev, command: str) -> str:
    print(f"?? {command}")
    response = dev.query(command).strip()
    print(f"<< {response}")
    return response


def parse_scalar(raw: str) -> float | None:
    text = str(raw).strip()
    if not text:
        return None
    return float(text.split(",")[0].strip())


def main() -> None:
    args = parse_args()
    rm = pyvisa.ResourceManager()
    dev = rm.open_resource(args.address)
    dev.timeout = 10000
    dev.read_termination = "\n"
    dev.write_termination = "\n"

    suffix = str(args.channel)
    channel_list = f"(@{args.channel})"

    try:
        print(f"address = {args.address}")
        print(f"channel = {args.channel}")
        query(dev, "*IDN?")

        try:
            query(dev, ":ROUT:TERM?")
        except Exception:
            print("terminal query not supported before setup")

        write(dev, "*RST")
        write(dev, "*CLS")
        try:
            write(dev, ":ROUT:TERM FRON")
        except Exception as exc:
            print(f"front-terminal command failed: {exc}")

        try:
            query(dev, ":ROUT:TERM?")
        except Exception:
            print("terminal query not supported after setup")

        write(dev, f":SOUR{suffix}:FUNC:MODE VOLT")
        write(dev, f':SENS{suffix}:FUNC "CURR","VOLT"')
        write(dev, f":SENS{suffix}:CURR:PROT {args.compliance:.12g}")
        write(dev, f":SOUR{suffix}:VOLT:RANG:AUTO ON")

        if args.auto_current_range:
            write(dev, f":SENS{suffix}:CURR:RANG:AUTO ON")
        else:
            write(dev, f":SENS{suffix}:CURR:RANG:AUTO OFF")
            write(dev, f":SENS{suffix}:CURR:RANG {args.current_range:.12g}")

        write(dev, f":SOUR{suffix}:VOLT {args.voltage:.12g}")
        write(dev, f":OUTP{suffix} ON")

        if args.settle > 0:
            time.sleep(args.settle)

        v_meas = parse_scalar(query(dev, f":MEAS:VOLT? {channel_list}"))
        i_meas = parse_scalar(query(dev, f":MEAS:CURR? {channel_list}"))

        print()
        print(f"V_cmd  = {args.voltage}")
        print(f"V_meas = {v_meas}")
        print(f"I_meas = {i_meas}")
        if i_meas not in (None, 0.0):
            print(f"R_est  = {args.voltage / i_meas}")
    finally:
        try:
            write(dev, f":OUTP{suffix} OFF")
        except Exception:
            pass
        try:
            write(dev, ":ABOR")
        except Exception:
            pass
        try:
            write(dev, f":OUTP{suffix} OFF")
        except Exception:
            pass
        dev.close()


if __name__ == "__main__":
    main()
