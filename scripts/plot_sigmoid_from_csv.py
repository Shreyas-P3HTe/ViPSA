#!/usr/bin/env python3
"""
Rebuild a voltage sigmoid PNG from a measured pulse CSV.

This matches the same inference I used manually:
1. Compress consecutive equal-voltage samples into segments.
2. Ignore zero-voltage gap segments.
3. Interpret the remaining pattern as repeated:
   set -> read -> reset -> read
4. Mark a loop as "switched" when:
   I_read_after_set > I_read_after_reset
5. Plot switching probability versus set voltage.

The script prefers Matplotlib when available so the PNG includes labels.
If Matplotlib is not installed, it falls back to a simple dependency-free
PNG renderer that still produces the curve.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import struct
import sys
import zlib
from collections import Counter, defaultdict


def _approx_equal(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def load_measurements(csv_path: str) -> list[tuple[float, float]]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])

        voltage_key = None
        for candidate in ("V_cmd (V)", "Voltage (V)", "V_cmd", "Voltage"):
            if candidate in fieldnames:
                voltage_key = candidate
                break
        if voltage_key is None:
            raise ValueError("Could not find a voltage column in the CSV.")

        current_key = None
        for candidate in ("Current (A)", "Current", "I_meas (A)"):
            if candidate in fieldnames:
                current_key = candidate
                break
        if current_key is None:
            raise ValueError("Could not find a current column in the CSV.")

        rows: list[tuple[float, float]] = []
        for row in reader:
            try:
                voltage = float(row[voltage_key])
                current = abs(float(row[current_key]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Could not parse row: {row}") from exc
            rows.append((voltage, current))

    if not rows:
        raise ValueError("CSV does not contain any data rows.")
    return rows


def compress_voltage_segments(
    rows: list[tuple[float, float]],
    tolerance: float = 1e-12,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    active_voltage: float | None = None
    active_currents: list[float] = []

    for voltage, current in rows:
        if active_voltage is None:
            active_voltage = voltage
            active_currents = [current]
            continue

        if _approx_equal(voltage, active_voltage, tolerance=tolerance):
            active_currents.append(current)
            continue

        segments.append((float(active_voltage), float(statistics.mean(active_currents))))
        active_voltage = voltage
        active_currents = [current]

    if active_voltage is not None:
        segments.append((float(active_voltage), float(statistics.mean(active_currents))))

    return segments


def infer_read_and_reset_voltage(
    nonzero_segments: list[tuple[float, float]],
) -> tuple[float, float]:
    voltage_counts = Counter(round(voltage, 12) for voltage, _current in nonzero_segments)
    positive_counts = {voltage: count for voltage, count in voltage_counts.items() if voltage > 0}
    negative_counts = {voltage: count for voltage, count in voltage_counts.items() if voltage < 0}

    if not positive_counts:
        raise ValueError("Could not infer a positive read voltage from the CSV.")
    if not negative_counts:
        raise ValueError("Could not infer a negative reset voltage from the CSV.")

    read_voltage = max(positive_counts.items(), key=lambda item: item[1])[0]
    reset_voltage = max(negative_counts.items(), key=lambda item: item[1])[0]
    return float(read_voltage), float(reset_voltage)


def extract_loops(
    segments: list[tuple[float, float]],
    read_voltage: float | None = None,
    reset_voltage: float | None = None,
) -> tuple[list[dict[str, float | int]], float, float]:
    nonzero_segments = [
        (voltage, current)
        for voltage, current in segments
        if abs(float(voltage)) > 1e-12
    ]
    if len(nonzero_segments) < 4:
        raise ValueError("Not enough non-zero voltage segments to reconstruct sigmoid loops.")

    if read_voltage is None or reset_voltage is None:
        inferred_read_voltage, inferred_reset_voltage = infer_read_and_reset_voltage(nonzero_segments)
        read_voltage = inferred_read_voltage if read_voltage is None else read_voltage
        reset_voltage = inferred_reset_voltage if reset_voltage is None else reset_voltage

    loops: list[dict[str, float | int]] = []
    cursor = 0
    skipped_segments = 0

    while cursor + 3 < len(nonzero_segments):
        set_segment = nonzero_segments[cursor]
        read_after_set_segment = nonzero_segments[cursor + 1]
        reset_segment = nonzero_segments[cursor + 2]
        read_after_reset_segment = nonzero_segments[cursor + 3]

        if (
            _approx_equal(read_after_set_segment[0], read_voltage)
            and _approx_equal(reset_segment[0], reset_voltage)
            and _approx_equal(read_after_reset_segment[0], read_voltage)
        ):
            read_after_set_current = float(read_after_set_segment[1])
            read_after_reset_current = float(read_after_reset_segment[1])
            loops.append(
                {
                    "set_voltage": float(set_segment[0]),
                    "read_after_set_current": read_after_set_current,
                    "read_after_reset_current": read_after_reset_current,
                    "switched": 1 if read_after_set_current > read_after_reset_current else 0,
                }
            )
            cursor += 4
            continue

        skipped_segments += 1
        cursor += 1

    if not loops:
        raise ValueError(
            "Could not reconstruct any set-read-reset-read loops. "
            "Check whether this CSV matches the expected sigmoid pulse format."
        )

    if skipped_segments:
        print(
            f"Warning: skipped {skipped_segments} non-matching segment positions while resynchronizing loops.",
            file=sys.stderr,
        )

    return loops, float(read_voltage), float(reset_voltage)


def summarize_loops(loops: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    grouped: defaultdict[float, list[dict[str, float | int]]] = defaultdict(list)
    for loop in loops:
        grouped[round(float(loop["set_voltage"]), 12)].append(loop)

    summary: list[dict[str, float | int]] = []
    for bucket in sorted(grouped):
        items = grouped[bucket]
        total_loops = len(items)
        switched_count = sum(int(item["switched"]) for item in items)
        summary.append(
            {
                "set_voltage": sum(float(item["set_voltage"]) for item in items) / total_loops,
                "switching_probability": switched_count / total_loops,
                "switched_count": switched_count,
                "total_loops": total_loops,
                "mean_read_after_set_current_a": (
                    sum(float(item["read_after_set_current"]) for item in items) / total_loops
                ),
                "mean_read_after_reset_current_a": (
                    sum(float(item["read_after_reset_current"]) for item in items) / total_loops
                ),
            }
        )

    return summary


def write_summary_csv(summary: list[dict[str, float | int]], summary_path: str) -> None:
    if not summary:
        return
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def save_plot_with_matplotlib(
    summary: list[dict[str, float | int]],
    output_path: str,
    title: str,
) -> bool:
    os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib_plot_sigmoid"))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    x_values = [float(row["set_voltage"]) for row in summary]
    y_values = [float(row["switching_probability"]) for row in summary]

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    axis.plot(x_values, y_values, marker="o", linewidth=1.8, color="#1f77b4")
    axis.set_xlabel("Set voltage (V)")
    axis.set_ylabel("Switching probability")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.35)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return True


def _draw_line(
    image: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    width: int = 1,
) -> None:
    def set_pixel(x: int, y: int) -> None:
        if 0 <= y < len(image) and 0 <= x < len(image[0]):
            image[y][x] = color

    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        for offset_x in range(-(width // 2), width // 2 + 1):
            for offset_y in range(-(width // 2), width // 2 + 1):
                set_pixel(x0 + offset_x, y0 + offset_y)
        if x0 == x1 and y0 == y1:
            break
        err2 = 2 * err
        if err2 >= dy:
            err += dy
            x0 += sx
        if err2 <= dx:
            err += dx
            y0 += sy


def _draw_circle(
    image: list[list[tuple[int, int, int]]],
    center_x: int,
    center_y: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    for y in range(center_y - radius - 1, center_y + radius + 2):
        for x in range(center_x - radius - 1, center_x + radius + 2):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                if 0 <= y < len(image) and 0 <= x < len(image[0]):
                    image[y][x] = color


def save_plot_without_dependencies(
    summary: list[dict[str, float | int]],
    output_path: str,
) -> None:
    width = 1000
    height = 620
    left = 100
    right = 40
    top = 50
    bottom = 90
    plot_width = width - left - right
    plot_height = height - top - bottom

    image = [[(255, 255, 255) for _x in range(width)] for _y in range(height)]
    x_values = [float(row["set_voltage"]) for row in summary]
    y_values = [float(row["switching_probability"]) for row in summary]
    x_min = min(x_values)
    x_max = max(x_values)

    def map_x(value: float) -> int:
        if x_max <= x_min:
            return left + plot_width // 2
        return int(round(left + ((value - x_min) / (x_max - x_min)) * plot_width))

    def map_y(value: float) -> int:
        return int(round(top + (1.0 - value) * plot_height))

    grid_color = (230, 230, 230)
    axis_color = (0, 0, 0)
    line_color = (31, 119, 180)

    for y_tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = map_y(y_tick)
        _draw_line(image, left, y, width - right, y, grid_color, width=1)
    for x_value in x_values:
        x = map_x(x_value)
        _draw_line(image, x, top, x, height - bottom, (240, 240, 240), width=1)

    _draw_line(image, left, top, left, height - bottom, axis_color, width=2)
    _draw_line(image, left, height - bottom, width - right, height - bottom, axis_color, width=2)

    points = [(map_x(x_value), map_y(y_value)) for x_value, y_value in zip(x_values, y_values)]
    for start_point, end_point in zip(points, points[1:]):
        _draw_line(image, start_point[0], start_point[1], end_point[0], end_point[1], line_color, width=3)
    for x, y in points:
        _draw_circle(image, x, y, 6, line_color)

    raw_scanlines = bytearray()
    for row in image:
        raw_scanlines.append(0)
        for red, green, blue in row:
            raw_scanlines.extend((red, green, blue))

    compressed = zlib.compress(bytes(raw_scanlines), 9)

    def png_chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    with open(output_path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(
            png_chunk(
                b"IHDR",
                struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0),
            )
        )
        handle.write(png_chunk(b"IDAT", compressed))
        handle.write(png_chunk(b"IEND", b""))


def save_sigmoid_png(
    summary: list[dict[str, float | int]],
    output_path: str,
    title: str,
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if save_plot_with_matplotlib(summary, output_path, title):
        return "matplotlib"

    save_plot_without_dependencies(summary, output_path)
    return "fallback"


def default_output_path(csv_path: str) -> str:
    root, _ext = os.path.splitext(csv_path)
    return f"{root}_sigmoid.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Measured pulse CSV to analyze.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PNG path. Defaults to <input>_sigmoid.png",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional CSV path for the extracted sigmoid summary.",
    )
    parser.add_argument(
        "--read-voltage",
        type=float,
        default=None,
        help="Override the inferred read voltage.",
    )
    parser.add_argument(
        "--reset-voltage",
        type=float,
        default=None,
        help="Override the inferred reset voltage.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output or default_output_path(args.csv_path)
    title = f"{os.path.basename(args.csv_path)}: switching probability vs set voltage"

    rows = load_measurements(args.csv_path)
    segments = compress_voltage_segments(rows)
    loops, read_voltage, reset_voltage = extract_loops(
        segments,
        read_voltage=args.read_voltage,
        reset_voltage=args.reset_voltage,
    )
    summary = summarize_loops(loops)

    backend = save_sigmoid_png(summary, output_path, title)
    if args.summary:
        write_summary_csv(summary, args.summary)

    print(f"Input CSV: {args.csv_path}")
    print(f"Output PNG: {output_path}")
    if args.summary:
        print(f"Summary CSV: {args.summary}")
    print(f"Read voltage: {read_voltage:.6g} V")
    print(f"Reset voltage: {reset_voltage:.6g} V")
    print(f"Loops reconstructed: {len(loops)}")
    print(f"PNG backend: {backend}")
    print()
    print("Set voltage, switching probability, switched_count/total_loops")
    for row in summary:
        print(
            f"{float(row['set_voltage']):.6g}, "
            f"{float(row['switching_probability']):.4f}, "
            f"{int(row['switched_count'])}/{int(row['total_loops'])}"
        )

    if backend != "matplotlib":
        print(
            "\nNote: matplotlib was not available, so the script used a "
            "dependency-free PNG fallback without text labels.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
