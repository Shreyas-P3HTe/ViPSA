# Step 1 Audit Findings

Scope for this pass:
- Audited the hardware-facing paths used by `vipsa/gui/Viewfinder4_tk.py`.
- Added a simulation path so the GUI backend can run without physical instruments.
- Did not fold the findings below into broader production fixes yet.

Validation notes:
- Mocked both SMUs, the 707B switch, Arduino stage, Zaber axes, and lights.
- Exercised `run_single_DCIV`, `run_single_pulse`, and a mixed-SMU protocol run successfully against the mocks.
- I could not launch a real `tk.Tk()` window in this shell because there is no display server, so GUI validation here is backend/import/workflow-level rather than interactive-window-level.

## Findings

### 1. High: real connection flow ignores the GUI's selected ports and the detected SMU addresses

Refs:
- `vipsa/gui/Viewfinder4_tk.py:804`
- `vipsa/gui/Viewfinder4_tk.py:824`
- `vipsa/gui/Viewfinder4_tk.py:1437`
- `vipsa/gui/Viewfinder4_tk.py:1479`
- `vipsa/workflows/Main4.py:116`
- `vipsa/workflows/Main4.py:159`

What happens:
- The GUI collects Arduino and Zaber port values from the user.
- The GUI also auto-detects VISA resources for the switch, Keithley, and Keysight.
- But `_connect_all()` still calls `self.vipsa.connect_equipment(SMU_name=selected_smu)`.
- `connect_equipment()` hard-codes `COM5`, `COM7`, and constructs `KeysightSMU(0)` / `KeithleySMU(0)` instead of using the detected VISA addresses.

Impact:
- Real startup can bind the wrong instrument, fail on systems that do not use the author's default COM ports, or succeed partially and then reconnect a different hardware topology than the GUI thinks it connected.

### 2. High: the default DCIV sweep path is a directory, but the code only validates `exists()`

Refs:
- `vipsa/gui/Viewfinder4_tk.py:40`
- `vipsa/gui/Viewfinder4_tk.py:41`
- `vipsa/gui/Viewfinder4_tk.py:989`
- `vipsa/gui/Viewfinder4_tk.py:1657`
- `vipsa/gui/Viewfinder4_tk.py:1775`
- `vipsa/workflows/Main4.py:662`
- `vipsa/workflows/Main4.py:1035`

What happens:
- `DEFAULT_SWEEP_PATH` points to the `"sweep patterns"` directory, not to a CSV file.
- The single-measurement and grid-measurement UIs use that value as the default sweep path.
- `_run_single_dciv_meas()` and `_run_grid_measurement()` only check `os.path.exists(...)`, so a directory passes validation.
- The workflow later hands that path to `pd.read_csv(...)`, which will fail once the measurement starts.

Impact:
- A default, untouched DCIV configuration can fail immediately at runtime even though the GUI accepted it as valid.

### 3. High: plotting is triggered from a background worker thread

Refs:
- `vipsa/gui/Viewfinder4_tk.py:529`
- `vipsa/gui/Viewfinder4_tk.py:562`
- `vipsa/gui/Viewfinder4_tk.py:996`
- `vipsa/gui/Viewfinder4_tk.py:1005`
- `vipsa/workflows/Main4.py:748`
- `vipsa/workflows/Main4.py:758`
- `vipsa/workflows/Main4.py:966`
- `vipsa/workflows/Main4.py:968`

What happens:
- Measurements run inside `_run_in_background()`, which uses a worker thread.
- The UI defaults `Plot Result` to `True` for both DCIV and pulse runs.
- The workflow then calls `Data_Handler.show_plot(...)`, `Data_Handler.show_pulse(...)`, and `plt.show()` from that worker thread.

Impact:
- Tk/Matplotlib UI work from a non-main thread can hang, deadlock, or crash unpredictably depending on backend and platform.

### 4. High: hardware access is unsynchronized across the worker thread, the monitor timer, and the abort button

Refs:
- `vipsa/gui/Viewfinder4_tk.py:287`
- `vipsa/gui/Viewfinder4_tk.py:342`
- `vipsa/gui/Viewfinder4_tk.py:529`
- `vipsa/gui/Viewfinder4_tk.py:562`
- `vipsa/gui/Viewfinder4_tk.py:1130`
- `vipsa/gui/Viewfinder4_tk.py:1189`
- `vipsa/gui/Viewfinder4_tk.py:1154`
- `vipsa/gui/Viewfinder4_tk.py:1164`

What happens:
- The worker thread drives SMUs, stage motion, and switching.
- The setup monitor polls `get_closed_channels()` every second from the UI thread.
- The abort button calls `_safe_stop_hardware()` from the UI thread and can open all 707B channels while the worker is still mid-operation.
- There is no mutex, serialized command queue, or dedicated hardware thread protecting shared instrument access.

Impact:
- Relay state, SMU output state, and monitor reads can interleave in unsafe ways.
- This is the main race-condition hotspot in the current design.

### 5. High: grid DCIV assumes one hard-coded 16-device measurement order and crashes on other grids

Refs:
- `vipsa/workflows/Main4.py:1035`
- `vipsa/workflows/Main4.py:1079`

What happens:
- `measure_IV_gridwise()` loads the CSV, then sorts and indexes device IDs against a fixed `meas_seq` list.
- Any grid file whose device IDs are not in that exact 16-item sequence will raise `ValueError` at `meas_seq.index(...)`.

Impact:
- Grid measurement is not general-purpose right now; it is tightly coupled to one specific layout and can crash before touching hardware.

### 6. Medium: cleanup after partial connection failure is brittle and can mask the original error

Refs:
- `vipsa/workflows/Main4.py:148`
- `vipsa/workflows/Main4.py:159`
- `vipsa/workflows/Main4.py:161`
- `vipsa/workflows/Main4.py:182`

What happens:
- The exception path in `connect_equipment()` and the logic in `disconnect_equipment()` assume objects like `self.top_light`, `self.stage`, and `self.Zaber` are already present.
- During a partial startup failure, cleanup can throw secondary exceptions or return ambiguous equipment state.

Impact:
- The first, useful error can get buried under cleanup noise, which makes failed bring-up harder to diagnose and increases the chance of leaving hardware in an unknown state.

## Simulation Deliverable

Simulation support was added for this step so the GUI backend can run without physical instruments:
- `vipsa/gui/Viewfinder4_tk.py`
- `vipsa/hardware/mock_hardware.py`

This simulation work is intentionally limited to enabling mock execution and audit coverage. The production issues listed above remain open for later steps.
