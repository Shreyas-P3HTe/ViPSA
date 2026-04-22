# ViPSA for_PUDA Branch

This branch is the PUDA-facing ViPSA runtime branch. It keeps the cleaned package layout from the stable repository, but carries the newer tkinter GUI path, PyVISA hardware drivers, and compatibility wrappers needed for the current PUDA workflow.

## What This Branch Contains

- Active tkinter GUIs for Viewfinder, standalone SMU control, crossbar control, protocol editing, and Testmaker.
- Canonical PyVISA drivers for the Keithley 2450, Keysight B2902B, and Keithley 707B switch matrix.
- Split movement-control modules for the Arduino stage, Zaber stages, and light control.
- Backward-compatible interfaces in `Movement.py`, `Source_Measure_Unit.py`, and `Vision.py` so existing workflow and GUI imports keep working while the implementation lives in smaller driver modules.
- Emulated GUI compatibility tooling and saved test artifacts under `archive/simulation_2026-04-15/`.

## Layout

- `vipsa/analysis/`: data handling, plotting helpers, sweep generation, and vision utilities
- `vipsa/gui/`: active tkinter GUIs and GUI-side protocol tools
- `vipsa/hardware/`: PyVISA drivers, stage/light drivers, switch control, and legacy compatibility layers
- `vipsa/workflows/`: measurement orchestration backends used by the GUIs
- `scripts/`: runnable helper scripts, manual workflows, and emulated GUI test harnesses
- `tests/`: regression and compatibility tests for the active runtime path
- `docs/`: active reference documents kept with the branch
- `archive/`: legacy files, audit docs, examples, simulation outputs, and historical material

## Running The GUIs

Run from the repository root. The launcher scripts set a writable Matplotlib config directory before starting the GUI:

```bash
./run_viewfinder.sh
./run_crossbar_tk.sh
./run_standalone_smu_tk.sh
./run_testmaker.sh
```

The same applications can also be launched as package modules:

```bash
python3 -m vipsa.gui.Viewfinder4_tk
python3 -m vipsa.gui.GUI_crossbar_tk
python3 -m vipsa.gui.GUI_standalone_SMU_tk
python3 -m vipsa.gui.Testmaker_tk
```

GUI apps need to be launched from a desktop session with display access. In sandboxed shells, Tk may fail with `couldn't connect to display ":0"` even when the code imports correctly.

## Hardware Driver Notes

The current PUDA branch uses the canonical driver module names:

- `vipsa/hardware/keithley_2450.py`
- `vipsa/hardware/keysight_b2902b.py`
- `vipsa/hardware/keithley_707b.py`
- `vipsa/hardware/stage.py`
- `vipsa/hardware/zaber.py`
- `vipsa/hardware/light.py`

Legacy imports are intentionally preserved through `vipsa.hardware.Source_Measure_Unit` and `vipsa.hardware.Movement`. Older workflow code can still import names such as `KeithleySMU`, `KeysightSMU`, `Keithley707B`, `SourceMeasureUnit`, `Stage`, `Zaber`, and `Light`, while new code should prefer the split driver modules directly.

The Keithley 707B low-level helpers now use switch-oriented names: `send_switch_command()` and `query_switch_expression()`. The older `write_tsp()` and `query_tsp()` names remain available as aliases for compatibility.

## Verification

Focused checks for the current driver split and compatibility layer:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_scpi_driver_commands.py tests/test_smu_backcompat_smoke.py
PYTHONDONTWRITEBYTECODE=1 python3 -m compileall vipsa scripts/emulated_viewfinder_e2e.py
```

The latest recorded GUI emulation run exercised a mixed `APPROACH`, `DCIV`, and `PULSE` protocol across simulated Keithley, Keysight, 707B, Arduino stage, Zaber stages, and light hardware. The saved artifacts live under `archive/simulation_2026-04-15/runs/live_gui_e2e_20260421_161121/`.

## Notes

- `for_PUDA` is currently the branch carrying the PUDA hardware-driver consolidation work.
- Internal imports should use package paths such as `vipsa.workflows.Main4` and `vipsa.hardware.keithley_707b`.
- Non-runtime material from earlier working folders is kept under `archive/` instead of being mixed into the active runtime tree.
- See `CHANGELOG.md` for the dated history of the GUI migration, PyVISA driver consolidation, and compatibility-test work.
- See `docs/MIGRATION_MAP.md` for old-to-new file mapping from the earlier workspace layout.
