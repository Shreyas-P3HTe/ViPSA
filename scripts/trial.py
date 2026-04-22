import pandas as pd

from vipsa.hardware.Source_Measure_Unit import KeithleySMU

# ---------------- USER INPUTS ----------------
CSV_PATH = "C:/Users/amdm/OneDrive - Nanyang Technological University/Backup/Desktop/sweep patterns/Sweep_5cycles_PEA.csv"
VTOL = 1e-9
# --------------------------------------------

def main():
    print("\n=== 4-WAY SPLIT SANITY CHECK ===\n")

    # Load sweep
    df = pd.read_csv(CSV_PATH, dtype=float)
    voltages = df.iloc[:, 1].to_numpy()

    print(f"Loaded sweep with {len(voltages)} points")
    print(f"Vmin = {voltages.min():.3f} V | Vmax = {voltages.max():.3f} V\n")

    # Init SMU object (no measurement performed)
    smu = KeithleySMU(0)

    # Run split
    splits = smu.split_sweep_by_4(voltages.tolist(), vtol=VTOL)

    # ---- summary ----
    print("Segments found:")
    for seg_no, tag, vlist, i0, i1 in splits:
        print(
            f"  Seg {seg_no:2d} | {tag:2s} | "
            f"points={len(vlist):4d} | "
            f"Vstart={vlist[0]: .3f} V â†’ Vend={vlist[-1]: .3f} V"
        )

    # ---- counts ----
    counts = {t: 0 for t in ["pf", "pb", "nf", "nb"]}
    for _, tag, _, _, _ in splits:
        counts[tag] += 1

    print("\nSegment counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # ---- expected pattern check ----
    tags = [tag for _, tag, _, _, _ in splits]
    print("\nFirst 12 tags:", tags[:12])

    print("\nSanity checks:")
    if max(counts.values()) - min(counts.values()) <= 1:
        print("  âœ” pf/pb/nf/nb counts roughly balanced")
    else:
        print("  âœ˜ imbalance detected (likely split bug)")

    if any(len(vlist) > 1500 for _, _, vlist, _, _ in splits):
        print("  âœ˜ found abnormally large segment (likely missed turning point)")
    else:
        print("  âœ” no runaway segments")

    print("\n=== END CHECK ===\n")


if __name__ == "__main__":
    main()
