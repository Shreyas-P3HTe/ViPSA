# Changelog

## 2026-04-20

### Current-probe measurement updates
- Added a new `CV_CURRENT_PROBE` measurement mode for applying a constant voltage over a fixed duration while sampling the resulting current over time.
- Added `constant_voltage_current_probe` support to both the Keysight and Keithley SMU wrappers so the new mode shares the same high-level API across mixed-SMU workflows.
- Added `run_constant_voltage_current_probe` to `Main4.py` and `run_single_current_probe` to `Main_crossbar.py` so the new mode can run through both the standalone and crossbar workflow layers.
- Added `CurrentProbe` artifact handling to `Datahandling.py`, including CSV normalization, metadata sidecars, saved plot generation, and a dedicated current-versus-time figure path.

### GUI and protocol updates
- Extended both protocol editors so `CV_CURRENT_PROBE` is available as a first-class protocol step with voltage, duration, sample-interval, compliance, SMU selection, and autorange controls.
- Added direct current-probe controls to the tkinter standalone SMU and crossbar GUIs.
- Added crossbar grid-action support for the new current-probe mode.
- Updated tkinter live-plot handling so streamed progress rows can switch the x-axis to time-based plotting for current-probe runs while preserving the existing voltage-based plotting for DCIV and pulse modes.

### Cleanup and verification
- Cleaned up the current-probe workflow wiring so progress callbacks now flow correctly through the main workflow and standalone GUI paths.
- Added a regression test covering the fake-SMU current-probe workflow, including saved CSV, PNG, metadata, and progress-callback behavior.
- Verified the touched modules with `compileall` and ran `python -m unittest tests.test_sweep_generation`, including the new current-probe regression coverage.
- Ran a local mock end-to-end simulation of the new current-probe mode through the direct workflow, protocol execution path, and crossbar wrapper without requiring attached hardware.

## 2026-04-18

### Merge and save-path fixes
- Merged the newer tkinter runtime pieces from `Nightly version` back into `vipsa_clean`, including `Viewfinder4_tk.py`, the tkinter `Testmaker`, the tkinter standalone SMU and crossbar frontends, crossbar workflow updates, launcher scripts, and GUI mapping docs.
- Hardened `Datahandling.py` so saved measurement rows remain flat when a backend returns NumPy object arrays of measurement dictionaries.
- Added recovery logic for malformed CSV rows that were previously marshaled as dict strings, so saved sweep plots can be rebuilt from already-bad files instead of showing empty traces.
- Added regression coverage for both the object-array save case and the malformed-CSV plot recovery case.

## 2026-04-15

### Audit and cleanup
- Finished the Viewfinder hardware audit pass and archived the detailed findings under `archive/simulation_2026-04-15/reports/`.
- Ran a full-window mock stress simulation with a mixed protocol, saved the run log and generated outputs under `archive/simulation_2026-04-15/runs/`, and recorded the summary in `final_audit_report.md`.
- Rolled the simulation-only GUI path back out of live `Viewfinder4_tk.py`, removing the mock import and simulation toggle from the production runtime.
- Archived the temporary mock backend and stress runner under `archive/simulation_2026-04-15/` so the production tree keeps the audit evidence without carrying test-only files at runtime.
- Hardened `Datahandling.py` artifact plot generation so worker-thread saves build headless figures instead of opening Matplotlib GUI figures during background runs.
- Removed generated cache directories and bytecode as part of the repository cleanup pass.

### GUI migration updates
- Split the `vipsa/gui` tree into active tkinter modules and archived PySimpleGUI modules so the maintained GUI path is now clearer.
- Moved the legacy PySimpleGUI GUI files into `archive/gui_psg/`, including `GUI_crossbar.py`, `GUI_standalone_SMU.py`, `GUI_standalone_SMU_2.py`, `ProtocolEditor.py`, `Viewfinder3.py`, and `Viewfinder4.py`.
- Refactored `Testmaker` onto a native tkinter/ttk implementation and removed its runtime dependency on PySimpleGUI.
- Added a new `GUI_standalone_SMU_tk.py` frontend for manual SMU measurements with list generation, protocol editing, autorange toggles, 4-way DCIV split control, live plotting, metadata context capture, and graph export.
- Added a new `GUI_crossbar_tk.py` frontend for crossbar selection and measurement with grid selection, single-device controls, protocol execution, autorange toggles, 4-way DCIV split control, live plotting, metadata context capture, and graph export.
- Reused the maintained `ProtocolBuilderTk` flow for the new tkinter GUIs instead of carrying forward the older PySimpleGUI protocol editor path.

### Crossbar backend updates
- Extended `Main_crossbar.py` so crossbar DCIV runs can use 4-way split sweeps when the selected SMU supports them.
- Threaded `current_autorange`, read-probe inclusion, read-probe mode, and live progress callbacks through the crossbar DCIV path.
- Extended the crossbar pulse path to accept `set_acquire_delay` and `current_autorange`.
- Added saved metadata sidecars for crossbar sweep, resistance, and pulse outputs so the new GUIs can persist operator notes and run context in the same artifact model as the newer workflow layer.

### Launch and environment updates
- Added `run_crossbar_tk.sh` and `run_standalone_smu_tk.sh` launchers, matching the existing `run_testmaker.sh` pattern by setting a writable `MPLCONFIGDIR` and launching from the repo root.
- Updated the README so the new tkinter launcher scripts are documented alongside `run_testmaker.sh`.
- Installed the missing runtime packages needed by the new tkinter GUI stack in the current environment, including `pandas`, `pyvisa`, `colorcet`, `scipy`, `pyserial`, and `pymeasure`.
- Verified that the new tkinter GUI modules compile with `py_compile` and that the launcher path for `GUI_crossbar_tk` reaches Tk startup, with the remaining launch constraint in this shell being desktop display access rather than missing Python dependencies.

## 2026-04-14

### GUI and workflow updates
- Added 2-way and 4-way split selection for DCIV flows and wired the selection through single, grid, and protocol execution paths.
- Added support for enabling or skipping the HRS/LRS read-probe during DCIV runs.
- Updated abort handling so an abort reconnects the Arduino stage before restoring the active SMU route.
- Added a dedicated `Settings` tab for runtime baseline values used by alignment, quick approach, single DCIV, single pulse, and grid runs.
- Added explicit defaults for those runtime settings and made them persist across app restarts in `C:\Users\shrey\.vipsa_control_center_settings.json`.
- Added autosave for settings changes on checkbox toggle, field focus-out, Enter key, and app close.
- Added persistent `Use Current Autorange` toggles in the Settings tab, Single DCIV, Single Pulse, and Grid measurement panels.
- Extended protocol step templates so DCIV and Pulse steps can carry a `current_autorange` setting.
- Added explicit mixed-SMU protocol handling so adjacent protocol steps can select different SMUs without manual reconnects between steps.
- Added GUI-side instrument detection that distinguishes the 707B matrix, Keithley SMU, and Keysight SMU and keeps the active standalone SMU aligned with the selected GUI connection.
- Added active-instrument switching in the tkinter control GUI so changing the SMU selection forces a clean disconnect/reconnect on both the USB side and the 707B routing side.
- Added a safe `ABORT` control to the tkinter GUI and kept the shutdown path focused on disabling outputs, reopening switch routes, and restoring stage access.
- Added a live 707B monitoring pane and active-setup block-diagram view in the tkinter GUI so the currently routed equipment path is visible while switching.
- Added live sweep plotting for DCIV-style runs in the tkinter GUI through the existing progress callback path.
- Consolidated the tkinter work onto the maintained `Viewfinder4_tk.py` path and fixed package-style imports so Spyder and direct module launches resolve the stable `vipsa_clean` package correctly.

### Measurement handling updates
- Changed SMU range selection to use the configured compliance current 1:1 when manual ranging is active.
- Added saved measurement graph images alongside exported CSV files for reference-only review, using the same base filename.
- Added start-polarity flexibility to the sweep listmaker so voltage lists can begin from positive or negative values as required.
- Cleaned up HRS/LRS read-probe handling so the read-probe path uses the intended compliance values more consistently.
- Centralized dataset artifact saving so one save path now owns the CSV, same-basename plot image, and same-basename metadata JSON sidecar.
- Added machine-readable run metadata for saved measurements, including timestamp, SMU details, compliance and timing settings, protocol context, and linked step information.
- Added `Cycle Number`, `V_cmd (V)`, `V_meas (V)`, and `V_error (V)` handling to the saved measurement pipeline, with graceful `NaN` fallback where a mode cannot provide reliable voltage readback.
- Added protocol-context metadata to per-step datasets so protocol runs record both the current step and the full list of sibling steps in the saved sidecar.
- Fixed the step-number save path so data files use the intended protocol/grid step index instead of accidentally reusing the z-height value.
- Fixed the empty resistance-save edge case so a missing or zero-length read-probe result no longer crashes the run after the main DCIV file is already saved.
- Fixed a post-save DCIV plotting regression where the legacy resistance viewer still assumed the old fixed CSV column order and could raise `unsupported operand type(s) for /: 'str' and 'str'` after a successful measurement save.
- Routed resistance post-plotting through the shared figure builder so richer saved datasets with `V_cmd`/`V_meas` columns remain viewable without breaking existing runs.

### Keithley and autorange updates
- Added a shared Keithley current-configuration helper that explicitly writes compliance with SCPI and only forces a manual current range when autorange is disabled.
- Threaded `current_autorange` through single DCIV, single pulse, grid, and protocol execution paths.
- Updated Keithley segmented sweeps, 4-way sweeps, read-probes, and pulsed measurements to respect the autorange toggle.
- Kept the new autorange flag compatible with the Keysight wrapper so mixed-SMU workflows continue to share the same high-level API.
- Fixed a remaining argument-passing mismatch where newer GUI or protocol paths could send `current_autorange` into an older stable `run_single_DCIV` signature.
- Added compatibility forwarding in the workflow layer so optional execution kwargs are only passed down to SMU methods that actually support them.
- Normalized the Keithley and Keysight high-level DCIV/pulse call shapes further so the shared frontend can use one parameter model across both vendors.
- Updated the Keysight list-sweep and pulse paths to return structured measurement rows that preserve commanded voltage, parsed readback voltage where available, current, and cycle information.
- Added non-fatal one-time warnings for measurement modes that do not expose reliable voltage readback, rather than silently inventing a value.

### Testmaker updates
- Added a standalone `Testmaker` GUI for building protocol-compatible test waveforms.
- Added a feasibility catalog covering supported and unsupported test families across basic electrical, memory, environmental, mechanism, neuromorphic, and p-bit/TRNG workflows.
- Added protocol-compatible DC sweep generation for standard I-V, forming, and variability-oriented sweep templates.
- Added protocol-compatible pulse generation for endurance, retention-style read trains, bias-stress, LTP/LTD, and PPF-style sequences using fixed-width slot timing.
- Added CSV preview/save flows and protocol JSON export in the current `{"type": ..., "params": ...}` schema without changing protocol runner semantics.
- Added full probability-voltage sigmoid collection export that saves one pulse CSV and one protocol JSON per bias voltage plus a manifest file for the whole sweep.
- Added an explicit sweep-direction mode selector for generated sweep lists so users can keep the existing positive-first pattern or generate negative-first and direct bipolar crossing patterns from the same shared generator.
- Refactored the stable sweep-list generation logic into a reusable shared module so the runtime listmaker and Testmaker use the same endpoint and cycle-number logic.

### Repository housekeeping
- Promoted the latest runtime modules into `vipsa_clean` as the cleaned stable repository baseline.
- Added `Testmaker.py` to the packaged GUI tree so it ships with the structured repository instead of living only as a loose top-level file.
- Sorted non-runtime material into `vipsa_clean/archive`, including legacy root files, audit docs, tests, examples, caches, and the machine-specific workflow copy.
- Created a sibling `Nightly version` clone from the cleaned stable repository for ongoing development work.
- Updated repository README files so the stable/nightly split and archive layout are documented.
- Compared the stable and nightly trees and merged the newer nightly tkinter GUI and Arduino reconnect helper back into `vipsa_clean`, making the stable tree the canonical merged runtime again.
- Saved the pre-merge stable copies of `Viewfinder4_tk.py` and `Openflexture.py` under `vipsa_clean/archive/merge_backup_2026-04-14/` before replacing them during consolidation.

### Documentation and tests
- Added `docs/PROTOCOL_FLOW.md` to describe the intended DCIV protocol pipeline and the explicit placement of HRS/LRS read-probes.
- Added regression tests for sweep generation modes and artifact sidecar creation in `vipsa_clean/tests/test_sweep_generation.py`.
- Verified the updated stable modules with `unittest` and `py_compile` under the base conda environment after the save-path, workflow, SMU, and sweep-generator changes.

### Notes
- GUI/runtime work was developed in the `Nightly version` tree, and the repository housekeeping pass also refreshed the stable `vipsa_clean` tree.
- Direct `python` and `py` launchers were not available on PATH in this shell, so verification was performed with `conda run -n base ...` instead.
