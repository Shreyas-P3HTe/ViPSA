# ViPSA Stable Repository

This repository is the cleaned, stable ViPSA codebase for running the machine and its supported tooling.

## Layout

- `vipsa/analysis/`: data handling, plotting helpers, and vision utilities
- `vipsa/gui/`: interactive GUIs, protocol editor, and `Testmaker`
- `vipsa/hardware/`: instrument, stage, and multiplexer control
- `vipsa/workflows/`: measurement orchestration backends
- `scripts/`: runnable helper scripts and manual workflows
- `docs/`: active reference documents kept with the stable repo
- `archive/`: legacy workspace files, audit docs, tests, examples, caches, and historical extras

## Running Modules

Run package modules from the repository root:

```powershell
python -m vipsa.gui.Viewfinder4
python -m vipsa.gui.GUI_standalone_SMU
python -m vipsa.gui.GUI_crossbar
python -m vipsa.gui.Testmaker
python -m scripts.grid_meas
```

For `Testmaker`, you can also use the repo launcher:

```bash
./run_testmaker.sh
```

Tk launchers are also available for the newer GUI rewrites:

```bash
./run_crossbar_tk.sh
./run_standalone_smu_tk.sh
./run_testmaker.sh
```

## Notes

- This folder is the stable baseline.
- A separate sibling copy named `Nightly version` is used for ongoing development.
- Internal imports use package paths such as `vipsa.workflows.Main4`.
- Non-runtime material from the earlier working folder was moved into `archive/` instead of being mixed into the stable runtime tree.
- GUI apps need to be launched from a desktop session with display access. In sandboxed shells, Tk may fail with `couldn't connect to display ":0"` even when the code is otherwise correct.

See `docs/MIGRATION_MAP.md` for the old-to-new file mapping.
