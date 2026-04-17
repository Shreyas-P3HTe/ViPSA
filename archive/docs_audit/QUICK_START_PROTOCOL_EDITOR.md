# Quick Start: Enhanced Protocol Editor

## What Changed?

The old protocol builder that only captured values from the Measure tab has been replaced with a **full-featured protocol editor** that opens a dialog where you can customize every parameter for each step.

## How to Use

### 1. Add a New Step

**Click**: `➕ Add Step` button in Protocol Builder tab

**What happens**:
1. Dialog opens with "Step Type" dropdown
2. Select a step type (DCIV, PULSE, ALIGN, APPROACH, RESISTANCE, DELAY, CUSTOM)
3. Parameter fields appear based on your selection
4. Fill in all parameters
5. Click "Add Step" or "Save & Close"

### 2. Edit an Existing Step

**Option A**: Select step in list, click `✏ Edit Selected`  
**Option B**: Double-click on step in list

**What happens**:
1. Dialog opens with current step's parameters
2. Edit any parameter
3. Click "Save & Close" to update
4. Protocol list refreshes

### 3. Remove a Step

**Click**: `🗑 Remove Selected` (after selecting step in list)

**Or**: Right-click on step (when implemented in future)

### 4. Save/Load Protocols

**Save**: `💾 Save Protocol`
- Prompts for filename
- Saves as JSON file
- Can be shared or used later

**Load**: `📂 Load Protocol`
- Choose JSON file
- Validates protocol structure
- Loads into builder

### 5. Run Protocol

**Click**: `▶ Run Protocol on Current Target` (green button)

**Requirements**:
- Equipment must be connected
- Sample ID and Device ID entered in Measure tab
- Save folder specified in Measure tab

---

## Step Types Explained

### 🔷 DCIV (DC/IV Sweep)

**What it does**: Sweeps voltage and measures current

**Parameters**:
- **Sweep Path** (file): CSV file with voltage sweep pattern
- **Pos Compliance** (A): Current compliance for positive voltages
- **Neg Compliance** (A): Current compliance for negative voltages
- **Sweep Delay** (s): Time between voltage steps
- **4-Way Split**: Use separate compliance for forward/backward sweeps?
- **Align**: Align device before measurement?
- **Approach**: Auto-approach before measurement?
- **SMU Select**: Which SMU to use

**Example**:
```
Sweep Path: sweep_patterns/sweep1V.csv
Pos Compliance: 0.001 A
Neg Compliance: 0.01 A
Sweep Delay: 0.0001 s
Align: YES
Approach: YES
```

### ⚡ PULSE (Pulsed Measurement)

**What it does**: Applies voltage pulse and measures response

**Parameters**:
- **Pulse Path** (file): CSV file with pulse voltages
- **Compliance** (A): Current compliance
- **Pulse Width** (s): Duration of each pulse
- **Acquire Delay** (s): When to measure (offset from pulse start)
- **Align**: Align device first?
- **Approach**: Auto-approach first?
- **SMU Select**: Which SMU to use

**Example**:
```
Pulse Path: pulse_patterns/pulse20.csv
Compliance: 0.01 A
Pulse Width: 0.001 s
Acquire Delay: 0.0005 s
Align: NO
Approach: YES
```

### 📍 ALIGN (Correct Course/Alignment)

**What it does**: Aligns imaging and corrects device misalignment

**Parameters**:
- **Move**: Actually move stage (or just photograph)?
- **Zaber Correction**: Use Zaber stages or OpenFlexture?
- **Recheck**: Take photo after correction?

**Example**:
```
Move: YES
Zaber Correction: YES
Recheck: YES
```

### ✋ APPROACH (Detect Contact & Approach)

**What it does**: Moves Z-stage until electrical contact is detected

**Parameters**:
- **Step Size** (um): How far to move per step
- **Test Voltage** (V): Small voltage to test contact
- **Lower Threshold** (A): Minimum current = contact
- **Upper Threshold** (A): Maximum safe current
- **Max Attempts**: Safety limit on Z movements
- **Delay** (s): Wait time between steps

**Example**:
```
Step Size: 0.5 um
Test Voltage: 0.1 V
Lower Threshold: 1e-11 A
Upper Threshold: 5e-11 A
Max Attempts: 50
Delay: 1 s
```

### 🔬 RESISTANCE (Measure Resistance)

**What it does**: Measures device resistance at fixed voltage

**Parameters**:
- **Voltage** (V): Voltage to apply
- **SMU Select**: Which SMU to use

### ⏳ DELAY (Wait)

**What it does**: Pause measurement sequence

**Parameters**:
- **Duration** (s): How long to wait

### 🔧 CUSTOM (Custom Sequence)

**What it does**: Run custom JSON parameters

**Parameters**:
- **JSON Params**: Raw JSON with custom configuration

**Example**:
```json
{
  "type": "custom_measurement",
  "samples": 100,
  "averaging": true
}
```

---

## Common Workflows

### Workflow 1: Simple IV Sweep

1. **Add Step**: ALIGN
   - Move: YES
   - Zaber Correction: YES
   - Recheck: YES

2. **Add Step**: APPROACH
   - All defaults OK
   - Adjust "Step Size" if needed

3. **Add Step**: DCIV
   - Sweep Path: Select your sweep pattern
   - Pos Compliance: 0.001 A
   - Neg Compliance: 0.01 A
   - Align: NO (already done)
   - Approach: NO (already done)

4. **Run** on single device

### Workflow 2: Pulsed Endurance Test

1. **Add Step**: ALIGN

2. **Add Step**: APPROACH

3. **Add Step**: PULSE
   - Pulse Path: pulse20.csv
   - Compliance: 0.01 A
   - Pulse Width: 0.001 s
   ```
   → Run pulses to set device state
   ```

4. **Add Step**: PULSE (again - lower compliance)
   - Pulse Path: pulse5.csv
   - Compliance: 0.001 A
   ```
   → Gentle read pulses to measure state
   ```

5. **Repeat**: Add more PULSE steps as needed

6. **Run** protocol on grid

### Workflow 3: Multi-SMU Test Sequence

**Future Enhancement** - Currently single SMU per step, but protocol can run on different SMUs:

1. **Add Step**: DCIV
   - SMU Select: Keithley2450
   - ...parameters...

2. **Add Step**: ALIGN (no SMU needed)

3. **Add Step**: PULSE
   - SMU Select: KeysightB2901BL (different SMU!)
   - ...parameters...

---

## Tips & Tricks

### Reuse Protocol Steps

**Copy-Paste via Save/Load**:
1. Create and save a protocol
2. Open it in text editor (it's JSON)
3. Copy-paste "steps" section
4. Use in another protocol file

### Debug Protocol Issues

1. **Run single device first**: Use "Run Protocol on Current Target"
   - See exactly what happens
   - Check output logs
   - Verify parameters

2. **Check parameter values**: Hover over protocol step to see full params

3. **Enable logging**: Check console output for step-by-step execution

### Performance Tuning

- **Sweep Delay**: Decrease for faster measurement, increase for stability
- **Pulse Width**: Longer = more charge, shorter = faster
- **Threshold Values**: Wider range = faster contact detection, narrower = more selective

---

## Parameter Ranges

**Safe Ranges** (to avoid hardware damage):

| Parameter | Min | Max | Unit | Notes |
|-----------|-----|-----|------|-------|
| Compliance | 0.001 | 10 | A | Set based on expected current |
| Voltage | -10 | +10 | V | SMU-dependent limit |
| Test Voltage | 0.01 | 1 | V | Should be < 15% of switching voltage |
| Threshold | 1e-12 | 1e-3 | A | Range of expected contact current |
| Sweep Delay | 0.0001 | 10 | s | Shorter = faster, longer = more stable |
| Pulse Width | 0.0001 | 1 | s | device-dependent |
| Step Size | 0.1 | 10 | um | Stage-dependent |

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| "Protocol is empty" | No steps added | Click "Add Step" first |
| "Equipment not connected" | Missing equipment | Run "Connect All Equipments" first |
| "Invalid parameters" | Out-of-range values | Check parameter ranges above |
| Dialog won't open | Click not registered | Click the button again slowly |
| Parameters not saved | Hit wrong button | Click "Save & Close", not "Add Step" |
| Old values still shown | Dialog cached | Close and reopen GUI |

---

## Future Enhancements

Planned additions:

- [ ] Loops and conditionals (IF/ELSE, FOR loops)
- [ ] Step failure handling (retry, skip, abort)
- [ ] Parameter linking (use output of step N as input to step N+1)
- [ ] Protocol templates/presets
- [ ] Graphical protocol flow editor
- [ ] Real-time step execution progress
- [ ] Parameter dependencies validation
- [ ] Measurement result display in protocol list

---

## Need Help?

1. **See documentation**: `FIXES_AND_RECOMMENDATIONS.md`
2. **View source**: `ProtocolEditor.py` (well-commented)
3. **Ask questions**: Review inline comments in Viewfinder4.py
4. **Check logs**: Console output shows step-by-step execution

---

**Version**: 1.0 (March 2026)  
**Last Updated**: 2026-03-04  
**Status**: Ready to Use ✅
