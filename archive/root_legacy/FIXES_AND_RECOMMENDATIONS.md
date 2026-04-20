# ViPSA Code - Fixes and Recommendations

## 🎯 Priority Fix Checklist

### 🔴 CRITICAL - Must Fix Immediately

#### 1. Missing Return Statements in Main4.py

**Problem**: Multiple methods have `return` instead of `return tuple_value`

**Files**: `Main4.py`

**Fix for `detect_contact_and_move_z()`**:
```python
# Line ~190-191: Change
finally:
    return  # WRONG

# To:
finally:
    return contact_detected, height, current
```

**Fix for `run_single_DCIV()`**:
```python
# Line ~560: Change
elif not contact : 
    # Contact detection failed
    return  # WRONG

# To:
elif not contact:
    print("Contact not established")
    return False, 0, None
```

**Action**: Add proper return tuples as documented in method docstrings

---

#### 2. Incomplete Method Bodies with `/* */` Markers

**Problem**: Code sections marked with C-style comments indicate incomplete Python

**Files**: `Main4.py`, `Source_Measure_Unit.py`

**Examples**:
- Line ~37-55 in `Main4.py::connect_equipment()` - connection logic cut off
- Line ~794-815 in `Source_Measure_Unit.py::pulsed_measurement()` - method body missing
- Line ~849-860 in `Source_Measure_Unit.py::split_pulse_for_2_chan()` - returns nothing

**Fix**: Uncomment/complete the methods or remove them entirely. For example:

```python
def split_pulse_for_2_chan(self, vlist):
    """Split voltage list into positive and negative channels."""
    vlist_p = [v for v in vlist if v >= 0]
    vlist_n = [v for v in vlist if v < 0]
    return vlist_p, vlist_n  # MUST return something
```

---

#### 3. Wrong Function Call in `best_5_endurance()`

**Problem**: Calls method with extra `self` argument

**File**: `Main4.py`, Line ~1059

**Current Code**:
```python
self.measure_IV_gridwise(self, sample_ID, gridpath, ...)  # WRONG - double self
```

**Fix**:
```python
self.measure_IV_gridwise(sample_ID, gridpath, ...)  # Correct
```

---

#### 4. Keithley707B Incomplete Implementation

**Problem**: SCPI commands don't check errors, no validation

**File**: `Source_Measure_Unit.py`

**Current Code**:
```python
def close_channel(self, channel):
    self.resource.write(":ROUT:CLOS (@<channels>)")  # No error checking
```

**Fix**:
```python
def close_channel(self, channel):
    """Close relay channel on switch matrix."""
    try:
        channels = self._format_channels(channel)
        self.resource.write(f":ROUT:CLOS (@{channels})")
        # Check for errors
        error = self.resource.query(":SYST:ERR?")
        if "+0," not in error:  # Error occurred
            print(f"Switch error: {error}")
            return False
        return True
    except Exception as e:
        print(f"Error closing switch channel: {e}")
        return False
```

---

### 🟠 HIGH Priority - Fix Soon

#### 5. VISA Resource Manager Memory Leak

**Problem**: ResourceManager created multiple times, never explicitly closed

**Files**: `Source_Measure_Unit.py` - ALL measurement methods

**Current Pattern**:
```python
def list_IV_sweep_manual(self, csv_path, pos_compliance, neg_compliance, delay=None, adr=None):
    if adr is None:
        adr = self.address
    
    rm = pyvisa.ResourceManager()  # Created here
    # ... method uses rm ...
    # But never explicitly closed!
```

**Fix**:
```python
class KeysightSMU:
    def __init__(self, device_no, address=None, ...):
        # ... existing code ...
        self.rm = pyvisa.ResourceManager()  # Create once, reuse
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'rm'):
                self.rm.close()
        except Exception:
            pass
    
    def list_IV_sweep_manual(self, csv_path, ...):
        # Use self.rm instead of creating new one
        try:
            smu = self.rm.open_resource(self.address)
            # ... rest of method ...
        finally:
            smu.close()
```

---

#### 6. No VISA Timeout Handling

**Problem**: SMU queries can hang indefinitely

**Files**: All SMU methods in `Source_Measure_Unit.py`

**Current Code**:
```python
def list_IV_sweep_manual(self, ...):
    smu = rm.open_resource(adr)
    # No timeout set - can hang forever on network issues
    current = float(SMU.query('READ?'))
```

**Fix**:
```python
def __init__(self, device_no, address=None, ...):
    # ...
    SMU = rm.open_resource(self.address)
    SMU.timeout = 10000  # Set 10 second timeout
    SMU.read_termination = '\n'
    SMU.write_termination = '\n'

def list_IV_sweep_manual(self, ...):
    try:
        smu = self.rm.open_resource(self.address)
        smu.timeout = 10000
        # Now queries will timeout instead of hanging
        current = float(smu.query('READ?'))
    except pyvisa.VisaIOError as e:
        print(f"VISA Timeout/Error: {e}")
        return None
```

---

#### 7. Input Validation Missing for Numeric Parameters

**Problem**: No validation that compliance/voltage/delay are positive and in valid range

**Files**: `Viewfinder4.py::_run_single_dciv_meas()`, grid methods, protocol execution

**Current Code**:
```python
pos_comp = float(values['-MEAS_POS_COMP-'])  # Could be negative, 0, or unreasonably large
SMU.run_single_DCIV(..., pos_compl=pos_comp, ...)  # Sent directly to hardware
```

**Fix**:
```python
def _validate_measurement_parameters(self, pos_compl, neg_compl, sweep_delay, compliance):
    """Validate measurement parameters are in valid ranges."""
    errors = []
    
    if pos_compl is not None and (pos_compl <= 0 or pos_compl > 10):  # Compliance should be < 10A
        errors.append(f"Positive compliance {pos_compl} out of range (0.001-10 A)")
    
    if neg_compl is not None and (neg_compl <= 0 or neg_compl > 10):
        errors.append(f"Negative compliance {neg_compl} out of range (0.001-10 A)")
    
    if sweep_delay is not None and (sweep_delay < 0 or sweep_delay > 10):
        errors.append(f"Sweep delay {sweep_delay} out of range (0-10 s)")
    
    return errors

def _run_single_dciv_meas(self, values):
    #... extract parameters ...
    errors = self._validate_measurement_parameters(pos_comp, neg_comp, sweep_delay, None)
    if errors:
        sg.popup_error('\n'.join(errors))
        return
    
    # Only proceed if validation passed
    self.vipsa.run_single_DCIV(...)
```

---

#### 8. Equipment Connection Not Verified Before Measurement

**Problem**: All measurement methods assume equipment is connected

**Files**: All measurement methods in `Viewfinder4.py` and `Main4.py`

**Current Code**:
```python
def _run_single_dciv_meas(self, values):
    if not self.is_equipment_connected:
        print("Error: Equipment not connected.")
        return
    # But what if connection was lost? No verification here
    self.vipsa.run_single_DCIV(...)  # Could crash silently
```

**Fix**:
```python
def _verify_equipment_connection(self):
    """Verify that all equipment is still connected."""
    errors = []
    
    if not hasattr(self.vipsa, 'SMU') or self.vipsa.SMU is None:
        errors.append("SMU not connected")
    else:
        try:
            # Query SMU identity
            idn = self.vipsa.SMU.get_address()
            print(f"SMU verified: {idn}")
        except Exception as e:
            errors.append(f"SMU connection lost: {e}")
    
    if self.vipsa.zaber_x is None or self.vipsa.zaber_y is None:
        errors.append("Zaber stages not connected")
    
    if self.vipsa.stage is None:
        errors.append("Arduino stage not connected")
    
    return errors

def _run_single_dciv_meas(self, values):
    if not self.is_equipment_connected:
        sg.popup_error("Equipment not connected")
        return
    
    # Re-verify connection hasn't been lost
    errors = self._verify_equipment_connection()
    if errors:
        sg.popup_error("Equipment connection issue:\n" + '\n'.join(errors))
        return
    
    # Now safe to proceed
    try:
        self.vipsa.run_single_DCIV(...)
    except Exception as e:
        sg.popup_error(f"Measurement failed: {e}")
        self.is_equipment_connected = False
```

---

### 🟡 MEDIUM Priority - Improve UX/Code Quality

#### 9. No Progress Feedback During Long Measurements

**Problem**: Grid measurements and protocols can run for minutes with no feedback

**Files**: `Viewfinder4.py`, `Main4.py`

**Solution**: Add progress bar or status updates

```python
def _run_grid_measurement(self, values, measurement_type='DCIV'):
    """Run grid measurement with progress feedback."""
    # ... parameter extraction ...
    
    grid_data = pd.read_csv(grid_path)
    device_coords = grid_data[['Device', 'X', 'Y']].values.tolist()
    
    # Create progress window (optional)
    total_devices = len(device_coords)
    
    for i, (dev_id, x, y) in enumerate(device_coords):
        # Skip logic ...
        
        # Update progress in main window log
        progress_pct = int((i / total_devices) * 100)
        print(f"[Progress {progress_pct}%] Device {int(dev_id)}/{len(device_coords)}")
        
        # Actual measurement...
        self.vipsa.run_protocol(...)
```

---

#### 10. Grid Measurement Can't Be Cancelled

**Problem**: Long-running grid measurements block UI and can't be stopped

**Files**: `Viewfinder4.py`, `Main4.py`

**Solution**: Add cancellation flag

```python
class VipsaGUI:
    def __init__(self):
        # ...
        self.measurement_cancel_flag = False
    
    def _run_grid_measurement(self, values, measurement_type='DCIV'):
        self.measurement_cancel_flag = False
        
        for i, (dev_id, x, y) in enumerate(device_coords):
            if self.measurement_cancel_flag:
                print("Grid measurement cancelled by user")
                break
            
            # Actual measurement...

# Add a Cancel button to UI when measurement is running
elif event == '-CANCEL_MEASUREMENT-':
    self.measurement_cancel_flag = True
    print("Cancellation requested...")
```

---

#### 11. File I/O Not Atomic

**Problem**: Data files can be corrupted if process crashes during write

**Solution**: Use atomic write pattern

```python
import tempfile
import shutil

def save_measurement_data(self, data, filepath):
    """Save data safely using atomic write."""
    try:
        # Create temp file in same directory as target
        temp_fd, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(filepath),
            prefix='.tmp_'
        )
        
        try:
            # Write to temp file
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(data, f)
            
            # Atomic replace
            shutil.move(temp_path, filepath)
            print(f"Data saved safely to: {filepath}")
            return True
        
        except Exception as e:
            os.close(temp_fd)
            os.unlink(temp_path)
            raise
    
    except Exception as e:
        print(f"Failed to save data: {e}")
        return False
```

---

#### 12. Logging Module Inconsistency

**Problem**: `log.error()`, `log.info()` called but logging not initialized

**Files**: `Source_Measure_Unit.py`

**Fix**:

Add at module top:
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)
```

Or remove log calls entirely and use `print()` + file logging in backend.

---

#### 13. Unused Import - Keithley2450

**Problem**: `from pymeasure.instruments.keithley import Keithley2450` but never used

**Files**: `Source_Measure_Unit.py`, Line 11

**Fix**: Either implement a Keithley2450 wrapper class or remove the import

```python
# Option 1: Implement wrapper
class Keithley2450SMU(Keithley2450):
    """Wrapper around pymeasure Keithley2450."""
    def __init__(self, gpib_address=12):
        super().__init__(f'GPIB0::{gpib_address}::INSTR')
    
    def measure_current_at_voltage(self, voltage):
        self.apply_voltage(voltage)
        return self.current

# Option 2: Remove unused import
# from pymeasure.instruments.keithley import Keithley2450  # DELETE THIS LINE
```

---

## 🔧 Implementation Order

### Week 1 - Critical Fixes
1. ✅ Add missing return statements to all measurement methods
2. ✅ Complete method bodies (remove `/* */` markers)
3. ✅ Fix `best_5_endurance()` duplicate self argument
4. ✅ Add VISA timeout handling across all SMU methods

### Week 2 - Resource Management
5. Implement ResourceManager as instance variable (not per-call)
6. Add `__del__` cleanup methods to SMU classes
7. Add connection verification before measurements

### Week 3 - Input Validation & Error Handling
8. Add numeric parameter validation
9. Add equipment connection verification
10. Implement centralized error handler/user notifications

### Week 4 - Usability Improvements
11. Add progress feedback for long measurements
12. Implement cancellation mechanism
13. Add atomic file I/O for data safety

---

## ✅ Newly Implemented Features

### Enhanced Protocol Editor (NEW!)

The new `ProtocolEditor.py` dramatically improves protocol building:

**Features**:
- ✅ Popup dialog for each protocol step with full parameter editing
- ✅ Support for DCIV, PULSE, ALIGN, APPROACH, RESISTANCE, DELAY, CUSTOM steps
- ✅ Dynamic form fields based on step type
- ✅ Parameter validation before addition
- ✅ Human-readable protocol display in listbox
- ✅ Edit existing steps by double-clicking
- ✅ Save/load protocols to JSON

**Usage**:
```python
# From Viewfinder4.py
self.protocol_builder = ProtocolBuilder(self.window, self.vipsa)

# User clicks "Add Step" button
# Dialog opens with all parameter inputs
# User fills in values for compliance, sweep_path, delays, etc.
# Dialog validates and adds to protocol_list_configs
```

---

## 📋 Code Review Checklist

Before committing changes, verify:

- [ ] All methods have return statements that match their docstrings
- [ ] No `/* */` C-style comment blocks in Python code
- [ ] VISA operations have timeout values set
- [ ] Numeric parameters validated before use
- [ ] Equipment state verified before operations
- [ ] Resource cleanup in `__del__` or context managers
- [ ] File I/O uses atomic write pattern
- [ ] Logging configured at module level
- [ ] Unused imports removed
- [ ] Error messages informative and user-facing
- [ ] Long operations show progress feedback
- [ ] All string formatting uses f-strings (not .format())
- [ ] Type hints on function signatures (where useful)

---

## 📚 Related Files

- `AUDIT_ISSUES.md` - Detailed issue catalog
- `ProtocolEditor.py` - New enhanced protocol editor (READY TO USE)
- `Viewfinder4.py` - Updated to use ProtocolEditor
- `Main4.py` - Needs return statements and method completions
- `Source_Measure_Unit.py` - Needs timeout, cleanup, validation additions
