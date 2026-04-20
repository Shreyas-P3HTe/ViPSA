"""
Enhanced Protocol Editor Dialog
Allows users to add complex measurement sequences with full parameter control.
Supports multiple test types, SMUs, and custom configurations.
"""

import PySimpleGUI as sg
import json
from copy import deepcopy


class ProtocolStepEditor:
    """Dialog for editing individual protocol steps with full parameter control."""
    
    # Define protocol step templates with default parameters
    STEP_TEMPLATES = {
        'DCIV': {
            'display_name': 'DC/IV Sweep',
            'params': {
                'sweep_path': '',
                'pos_compl': 0.001,
                'neg_compl': 0.01,
                'sweep_delay': 0.0001,
                'align': False,
                'approach': False,
                'smu_select': 'Keithley2450',
                'use_4way_split': True,
            }
        },
        'PULSE': {
            'display_name': 'Pulsed Measurement',
            'params': {
                'pulse_path': '',
                'compliance': 0.01,
                'pulse_width': 0.001,
                'align': False,
                'approach': False,
                'smu_select': 'Keithley2450',
                'set_acquire_delay': 0.0005,
            }
        },
        'CV_CURRENT_PROBE': {
            'display_name': 'Constant Voltage Current Probe',
            'params': {
                'voltage': 0.1,
                'duration': 1.0,
                'sample_interval': 0.1,
                'compliance': 0.001,
                'align': False,
                'approach': False,
                'smu_select': 'Keithley2450',
                'current_autorange': False,
            }
        },
        'ALIGN': {
            'display_name': 'Correct Course (Align)',
            'params': {
                'move': True,
                'zaber_corr': True,
                'recheck': True,
            }
        },
        'APPROACH': {
            'display_name': 'Detect Contact & Approach',
            'params': {
                'step_size': 0.5,
                'test_voltage': 0.1,
                'lower_threshold': 1e-11,
                'upper_threshold': 5e-11,
                'max_attempts': 50,
                'delay': 1,
            }
        },
        'CUSTOM': {
            'display_name': 'Custom Sequence',
            'params': {
                'description': 'Define custom JSON parameters',
                'json_params': '{}',
            }
        },
    }
    
    def __init__(self, parent_values=None, editing_index=None):
        """
        Initialize the editor.
        
        Args:
            parent_values: Current GUI values from parent window
            editing_index: If editing existing step, the index in protocol_list
        """
        self.parent_values = parent_values or {}
        self.editing_index = editing_index
        self.selected_type = None
        self.edited_config = None
        
    def create_layout(self, initial_step=None):
        """Create the protocol step editor layout."""
        
        sg.theme('DarkBlue')
        
        # Step type selection
        type_layout = [
            [sg.Text('Step Type:', font=('Arial', 10, 'bold'))],
            [sg.Combo(list(self.STEP_TEMPLATES.keys()) + ['RESISTANCE', 'DELAY', 'LOG_MESSAGE'],
                      default_value=initial_step.get('type', 'DCIV') if initial_step else 'DCIV',
                      key='-STEP_TYPE-',
                      readonly=True,
                      enable_events=True,
                      size=(30, 1))],
        ]
        
        # Parameter editing section (dynamic based on step type)
        param_layout = [
            [sg.Text('Parameters:', font=('Arial', 10, 'bold'))],
            [sg.Column([], key='-PARAM_COLUMN-', size=(500, 400), expand_x=True, expand_y=True)],
        ]
        
        # Bottom buttons
        button_layout = [
            [sg.Button('Add Step', key='-ADD_STEP-', size=(12, 1)),
             sg.Button('Cancel', key='-CANCEL-', size=(12, 1)),
             sg.Button('Save & Close', key='-SAVE_STEP-', size=(12, 1))]
        ]
        
        layout = [
            [sg.Text('Protocol Step Editor', font=('Arial', 14, 'bold'))],
            [sg.HSeparator()],
            [type_layout[0]],
            [type_layout[1]],
            [sg.HSeparator()],
            [param_layout[0]],
            [param_layout[1]],
            [sg.HSeparator()],
            [button_layout[0]],
        ]
        
        window = sg.Window('Add/Edit Protocol Step', layout, finalize=True, modal=True)
        
        # Set initial parameters if editing.  Always call `_update_param_fields`
        # so that the parameter column is populated with widgets; previously the
        # logic executed before the window was finalized and the update had no
        # effect, leaving the dialog blank.  Finalize is invoked above.
        if initial_step:
            self._populate_from_step(window, initial_step)
        else:
            self._update_param_fields(window, 'DCIV', self.STEP_TEMPLATES['DCIV'])
        
        return window
    
    def _create_param_fields(self, step_type):
        """Create input fields for the given step type."""
        template = self.STEP_TEMPLATES.get(step_type, {})
        params = template.get('params', {})
        
        field_layout = []
        
        for key, default_value in params.items():
            if isinstance(default_value, bool):
                field_layout.append([
                    sg.Checkbox(key.replace('_', ' ').title(), 
                               default=default_value, 
                               key=f'-PARAM_{key}-')
                ])
            elif isinstance(default_value, (int, float)):
                field_layout.append([
                    sg.Text(f'{key.replace("_", " ").title()}:', size=(20, 1)),
                    sg.InputText(str(default_value), key=f'-PARAM_{key}-', size=(20, 1))
                ])
            elif key in ('sweep_path', 'pulse_path'):
                field_layout.append([
                    sg.Text(f'{key.replace("_", " ").title()}:', size=(20, 1)),
                    sg.InputText(default_value, key=f'-PARAM_{key}-', size=(30, 1)),
                    sg.FileBrowse(size=(10, 1))
                ])
            elif key == 'smu_select':
                field_layout.append([
                    sg.Text('SMU:', size=(20, 1)),
                    sg.Combo(['Keithley2450', 'KeysightB2901BL'], 
                            default_value=default_value,
                            key=f'-PARAM_{key}-', 
                            readonly=True, 
                            size=(20, 1))
                ])
            elif key == 'json_params':
                field_layout.append([
                    sg.Text('JSON Parameters:', size=(20, 1))
                ])
                field_layout.append([
                    sg.Multiline(default_value, key=f'-PARAM_{key}-', size=(50, 10), expand_x=True)
                ])
            else:
                field_layout.append([
                    sg.Text(f'{key.replace("_", " ").title()}:', size=(20, 1)),
                    sg.InputText(str(default_value), key=f'-PARAM_{key}-', size=(20, 1))
                ])
        
        # Add note about parameters
        if field_layout:
            field_layout.insert(0, [sg.Text(f'Configure {step_type} parameters:', text_color='lightblue')])
        
        return field_layout
    
    def _update_param_fields(self, window, step_type, template):
        """Update the parameter fields dynamically.

        The column element identified by `-PARAM_COLUMN-` is a placeholder that we
        populate using :meth:`Window.extend_layout`.  The previous implementation
        attempted to call ``Column.update`` with a new layout; that API does not
        modify the layout and consequently the column stayed empty.  We now
        explicitly clear any existing ``-PARAM_`` elements and then append the
        new rows.
        """
        print(f"[DEBUG] _update_param_fields called for {step_type}")
        try:
            param_fields = self._create_param_fields(step_type)
            param_column = window['-PARAM_COLUMN-']

            # remove any existing parameter widgets so switching types doesn't
            # append to the old ones
            for key in list(window.key_dict.keys()):
                # only string keys are relevant; others are positional index numbers
                # ignore the column container itself, which uses the same prefix
                if isinstance(key, str) and key.startswith('-PARAM_') and key != '-PARAM_COLUMN-':
                    try:
                        elem = window[key]
                        # hide & destroy underlying widget to avoid duplicate
                        elem.update(visible=False)
                        try:
                            elem.Widget.destroy()
                        except Exception:
                            pass
                    except Exception:
                        pass

            # now add the new rows into the column
            window.extend_layout(param_column, param_fields)
            print(f"[DEBUG] added {len(param_fields)} param rows")
        except Exception as e:
            print(f"Error updating param fields: {e}")
    
    def _populate_from_step(self, window, step_config):
        """Populate editor fields from an existing step config."""
        step_type = step_config.get('type', 'DCIV')
        params = step_config.get('params', {})
        
        window['-STEP_TYPE-'].update(step_type)
        self._update_param_fields(window, step_type, self.STEP_TEMPLATES.get(step_type, {}))
        
        # Populate parameter values
        for key, value in params.items():
            field_key = f'-PARAM_{key}-'
            try:
                if isinstance(value, bool):
                    window[field_key].update(value)
                else:
                    window[field_key].update(str(value))
            except Exception:
                pass  # Field might not exist for this type
    
    def _extract_params(self, window, step_type):
        """Extract parameter values from window."""
        template = self.STEP_TEMPLATES.get(step_type, {})
        params = {}
        
        for key in template.get('params', {}).keys():
            field_key = f'-PARAM_{key}-'
            try:
                element = window[field_key]
                value = element.get()
                
                # Handle boolean fields (Checkboxes)
                if isinstance(element, sg.Checkbox):
                    params[key] = value
                # Handle JSON fields
                elif key == 'json_params':
                    try:
                        params[key] = json.loads(value) if value.strip() else {}
                    except json.JSONDecodeError:
                        params[key] = {}
                # Handle file path fields  
                elif key.endswith('_path'):
                    params[key] = value
                # Handle SMU selector
                elif key == 'smu_select':
                    params[key] = value
                # Handle boolean-named fields that are InputText  
                elif key in ('use_4way_split', 'move', 'zaber_corr', 'recheck', 'align', 'approach'):
                    if isinstance(element, sg.Checkbox):
                        params[key] = element.get()
                    else:
                        params[key] = value.lower() in ('true', '1', 'yes')
                # Handle numeric fields
                else:
                    try:
                        if '.' in str(value):
                            params[key] = float(value)
                        else:
                            params[key] = int(value)
                    except (ValueError, TypeError):
                        # If conversion fails, try to use as string
                        params[key] = value
            except KeyError:
                pass  # Field doesn't exist for this type
        
        return params
    
    def run(self, initial_step=None):
        """Run the editor dialog.

        This method loops on the window until the user accepts or cancels.  When
        the step type combobox changes we dismantle and recreate the window with
        the new type selected; this keeps the layout code simple and avoids the
        need to dynamically tear down widgets (which proved unreliable).
        """
        window = self.create_layout(initial_step)
        try:
            while True:
                event, values = window.read()
                if event == sg.WIN_CLOSED or event == '-CANCEL-':
                    break

                if event == '-STEP_TYPE-':
                    # rebuild dialog with newly chosen type
                    chosen = values['-STEP_TYPE-']
                    window.close()
                    window = self.create_layout({'type': chosen})
                    continue

                if event == '-ADD_STEP-' or event == '-SAVE_STEP-':
                    step_type = values['-STEP_TYPE-']
                    print(f"[DEBUG] Add/Save clicked, extracting {step_type}")
                    params = self._extract_params(window, step_type)
                    print(f"[DEBUG] extracted params: {params}")

                    # Validate params
                    if not self._validate_params(step_type, params):
                        sg.popup_error('Invalid parameters. Please check your input.')
                        continue

                    self.edited_config = {
                        'type': step_type,
                        'params': params
                    }
                    break
        finally:
            window.close()

        return self.edited_config
    
    def _validate_params(self, step_type, params):
        """Validate extracted parameters.

        Parameters may occasionally arrive as strings (e.g. when the dialog is
        first displayed or when a protocol is loaded from file); converting them
        to floats avoids ``'<=' not supported`` errors.  If conversion fails the
        comparison will raise and be caught below.
        """
        def num(v):
            try:
                return float(v)
            except Exception:
                return v

        try:
            if step_type in ('DCIV', 'PULSE', 'CV_CURRENT_PROBE'):
                # Validate numeric ranges
                if step_type == 'DCIV':
                    if num(params.get('pos_compl', 0)) <= 0 or num(params.get('neg_compl', 0)) <= 0:
                        return False
                    if num(params.get('sweep_delay', 0)) < 0:
                        return False
                if step_type == 'PULSE':
                    if num(params.get('compliance', 0)) <= 0:
                        return False
                    if num(params.get('pulse_width', 0)) <= 0:
                        return False
                if step_type == 'CV_CURRENT_PROBE':
                    if num(params.get('voltage', 0)) != num(params.get('voltage', 0)):
                        return False
                    if num(params.get('compliance', 0)) <= 0:
                        return False
                    if num(params.get('duration', 0)) < 0:
                        return False
                    if num(params.get('sample_interval', 0)) <= 0:
                        return False

            elif step_type == 'APPROACH':
                if num(params.get('step_size', 0)) <= 0:
                    return False
                if num(params.get('test_voltage', 0)) <= 0:
                    return False

            return True

        except Exception as e:
            print(f"Validation error: {e}")
            return False


class ProtocolBuilder:
    """Main protocol builder interface integrated into GUI."""
    
    def __init__(self, parent_window, vipsa_instance):
        """
        Initialize protocol builder.
        
        Args:
            parent_window: PySimpleGUI window object
            vipsa_instance: Vipsa_Methods instance for saving/loading
        """
        self.parent_window = parent_window
        self.vipsa = vipsa_instance
        self.protocol_list_configs = []
    
    def show_step_editor(self, initial_step=None, edit_index=None):
        """Show the protocol step editor dialog."""
        editor = ProtocolStepEditor(editing_index=edit_index)
        config = editor.run(initial_step)
        
        if config:
            if edit_index is not None:
                # Replace existing step
                self.protocol_list_configs[edit_index] = config
                return f"Updated: {config['type']}"
            else:
                # Add new step
                self.protocol_list_configs.append(config)
                return f"Added: {config['type']}"
        
        return None
    
    def get_protocol_display_list(self):
        """Get human-readable list of protocol steps for display."""
        display_list = []
        for i, step in enumerate(self.protocol_list_configs):
            step_type = step.get('type', 'UNKNOWN')
            params = step.get('params', {})
            
            # Create readable summary
            summary_parts = [f"{i+1}. {step_type}"]
            
            if step_type == 'DCIV':
                summary_parts.append(f"[Pos Compl: {params.get('pos_compl')} A]")
                if params.get('align'):
                    summary_parts.append("[ALIGN]")
                if params.get('approach'):
                    summary_parts.append("[APPROACH]")
            
            elif step_type == 'PULSE':
                summary_parts.append(f"[Compliance: {params.get('compliance')} A]")
                summary_parts.append(f"[Width: {params.get('pulse_width')} s]")

            elif step_type == 'CV_CURRENT_PROBE':
                summary_parts.append(f"[V: {params.get('voltage')} V]")
                summary_parts.append(f"[T: {params.get('duration')} s]")
                summary_parts.append(f"[dt: {params.get('sample_interval')} s]")
            
            elif step_type == 'APPROACH':
                summary_parts.append(f"[Threshold: {params.get('lower_threshold')}-{params.get('upper_threshold')}]")
            
            display_list.append(' '.join(summary_parts))
        
        return display_list
    
    def export_protocol(self, filepath):
        """Export protocol to JSON file."""
        try:
            return self.vipsa.save_protocol(filepath, self.protocol_list_configs)
        except Exception as e:
            sg.popup_error(f"Error saving protocol: {e}")
            return False
    
    def import_protocol(self, filepath):
        """Import protocol from JSON file."""
        try:
            proto = self.vipsa.load_protocol(filepath)
            ok, msg = self.vipsa.validate_protocol(proto)
            if not ok:
                sg.popup_error(f"Invalid protocol: {msg}")
                return False
            self.protocol_list_configs = proto
            return True
        except Exception as e:
            sg.popup_error(f"Error loading protocol: {e}")
            return False
    
    def clear_protocol(self):
        """Clear all steps from protocol."""
        self.protocol_list_configs = []
        return True
    
    def remove_step(self, index):
        """Remove a step at the given index."""
        if 0 <= index < len(self.protocol_list_configs):
            removed = self.protocol_list_configs.pop(index)
            return f"Removed: {removed['type']}"
        return None
