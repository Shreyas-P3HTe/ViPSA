# ViPSA - stable update 24/03/2026

This repository is a fresh, structured migration of the original ViPSA working folder.

## Layout

- `vipsa/analysis/`: data handling, plotting helpers, and vision utilities
- `vipsa/gui/`: interactive GUIs and protocol editor
- `vipsa/hardware/`: instrument, stage, and multiplexer control
- `vipsa/workflows/`: measurement orchestration backends
- `scripts/`: runnable helper scripts and manual workflows
- `examples/`: experiments and one-off prototypes kept for reference
- `docs/audit/`: audit and migration documentation from the original folder

## Running Modules

Run package modules from the repository root:

```powershell
python -m vipsa.gui.Viewfinder4
python -m vipsa.gui.GUI_standalone_SMU
python -m vipsa.gui.GUI_crossbar
python -m scripts.grid_meas
```

## Notes

- The original repository remains unchanged.
- Internal imports were rewritten to use package paths such as `vipsa.workflows.Main4`.
- Experimental files were preserved under `examples/` instead of being mixed into production code.

See `docs/MIGRATION_MAP.md` for the old-to-new file mapping.
