import PySimpleGUI as sg
import time # Keep time for potential delays
import os # Needed for path joining/checking

# Import the backend class
from Main_crossbar import Crossbar_Methods

class CrossbarGUI:
    def __init__(self):
        # Instantiate the backend methods class
        self.crossbar_methods = Crossbar_Methods()
        # No need for self.mux or self.ser here, managed by crossbar_methods

        self.window = self.create_window()
        self.grid_rects = {}
        self.selection_state = {}
        self.selected_device_count = 0
        self.protocol_list_configs = [] # Store full config dictionaries for the protocol

    def create_window(self):
        # --- Define default paths (adjust as needed) ---
        default_sweep_path = "C:/Users/amdm/Desktop/sweep patterns/Sweep_2Dmems_faster.csv"
        default_pulse_path = "C:/Users/amdm/Desktop/sweep patterns/Pulses_1000.csv"
        default_save_folder = "./crossbar_output" # Default save location

        sg.theme('Darkteal7') # Choose a theme

        ##################################### Grid Tab Layout #####################################################
        grid_layout = [
            [sg.Text("Multiplexer Grid View", justification='center', font='_ 14')],
            [sg.Text("Devices Selected: 0", key="-SELECTED_COUNT-", justification='center', font='_ 12')],
            [sg.Graph(canvas_size=(600, 600), graph_bottom_left=(-1, -1), graph_top_right=(17, 17),
                      key="-GRID-", enable_events=True, background_color='light gray', tooltip='Select devices to measure')]
        ]

        # Selection Buttons Layout for Grid Tab
        grid_buttons_layout = [
            [sg.Button("Select All", key="-SELECT_ALL-"),
             sg.Button("Deselect All", key="-DESELECT_ALL-")],
             [sg.Button("Select Half (Checkerboard)", key="-SELECT_HALF-"),
             sg.Button("Toggle Selection", key="-TOGGLE_SELECTION-")],
            [sg.Text("Default Action for Selected:")],
            [sg.Radio('Run DCIV', "GRID_ACTION", default=True, key="-GRID_DCIV-"),
             sg.Radio('Run Pulse', "GRID_ACTION", key="-GRID_PULSE-")],
            [sg.Button("Measure Selected Devices", key="-MEASURE_SELECTED-")]
        ]

        grid_column = sg.Column(grid_layout)
        buttons_column = sg.Column(grid_buttons_layout, vertical_alignment='top') # Align buttons nicely
        grid_tab_layout = [[grid_column, sg.VerticalSeparator(), buttons_column]] # Add separator

        ###################################### Individual Measurement Tab Layout #########################################
        # Section for Connection and Manual Channel Setting
        connection_layout = [
             [sg.Text("--- Connections ---", font='_ 11')],
             [sg.Button("Connect Equipments", key="-CONNECT-"), sg.Button("Disconnect Equipments", key="-DISCONNECT-")],
             [sg.Text("Channel 1 Pin (1-16):", size=(15, 1)), sg.InputText(key="-CH1-", size=(5, 1), default_text='1')],
             [sg.Text("Channel 2 Pin (1-16):", size=(15, 1)), sg.InputText(key="-CH2-", size=(5, 1), default_text='1')],
             [sg.Button("Set Manual Channels", key="-SET_CHANNELS-")],
        ]

        # Section for DCIV Measurement Parameters
        dciv_layout = [
            [sg.Text("--- DCIV Sweep ---", font='_ 11')],
            [sg.Text("Sweep Path:", size=(15,1)), sg.InputText(default_sweep_path, key='-SWEEP_PATH-', size=(50, 1)), sg.FileBrowse()],
            [sg.Text("Positive Comp (A):", size=(15,1)), sg.InputText(default_text='0.001', key='-DCIV_POS_COMP-', size=(10, 1)),
             sg.Text("Negative Comp (A):", size=(15,1)), sg.InputText(default_text='0.01', key='-DCIV_NEG_COMP-', size=(10, 1))],
            [sg.Text("Sweep Delay (s):", size=(15,1)), sg.InputText(default_text='0.0001', key='-DCIV_SWEEP_DELAY-', size=(10,1)),
             sg.Text("Acq. Delay (s, opt):", size=(15,1)), sg.InputText(key='-DCIV_ACQ_DELAY-', size=(10,1))], # Optional Acq Delay
             [sg.Checkbox('Plot Result', default=True, key='-DCIV_PLOT-'), sg.Checkbox('Save Result', default=True, key='-DCIV_SAVE-')],
             [sg.Button("Run Single DCIV", key='-RUN_SINGLE_DCIV-')],
        ]

        # Section for Pulsed Measurement Parameters
        pulse_layout = [
             [sg.Text("--- Pulsed Measurement ---", font='_ 11')],
             [sg.Text("Pulse Path:", size=(15,1)), sg.InputText(default_pulse_path, key='-PULSE_PATH-', size=(50, 1)), sg.FileBrowse()],
             [sg.Text("Compliance (A):", size=(15,1)), sg.InputText(default_text='0.01', key='-PULSE_COMP-', size=(10, 1))],
             [sg.Text("Pulse Width (s, opt):", size=(15,1)), sg.InputText(key='-PULSE_WIDTH-', size=(10,1))], # Optional width override
             [sg.Checkbox('Plot Result', default=True, key='-PULSE_PLOT-'), sg.Checkbox('Save Result', default=True, key='-PULSE_SAVE-')],
             [sg.Button("Run Single Pulse", key ='-RUN_SINGLE_PULSE-')]
        ]

        # General Save Location
        save_layout = [
            [sg.Text("--- Save Location ---", font='_ 11')],
             [sg.Text("Save Folder:", size=(15,1)), sg.InputText(default_save_folder, key='-SAVE_FOLDER-', size=(50, 1)), sg.FolderBrowse()],
        ]

        # Output Area
        output_layout = [
             [sg.Text("--- Output ---", font='_ 11')],
             [sg.Multiline(size=(80, 15), key="-OUTPUT-", autoscroll=True, disabled=True, reroute_stdout=True, reroute_stderr=True)], # Redirect print/errors
        ]

        ind_meas_layout = [
            [sg.Column(connection_layout)],
            [sg.HSeparator()],
            [sg.Column(dciv_layout)],
            [sg.HSeparator()],
            [sg.Column(pulse_layout)],
             [sg.HSeparator()],
             [sg.Column(save_layout)],
            [sg.HSeparator()],
            [sg.Column(output_layout)]
        ]

        ########################################### Protocol Tab Layout ##########################################
        protocol_editor_layout = [
            [sg.Text('Select Test Type:'), sg.Combo(['DCIV Sweep', 'Pulsed Measurement', 'Optimize Pulse'], key='-TEST_TYPE-', enable_events=True)],
            [sg.Button('Add Test to Protocol', key='-ADD_TEST-')],
            [sg.Button('Remove Selected Test', key='-REMOVE_TEST-')],
            [sg.Button('Clear Protocol', key='-CLEAR_PROTOCOL-')], # Added clear button
        ]
        protocol_viewer_layout = [
            [sg.Text("Protocol Steps:")],
            [sg.Listbox(values=[], size=(60, 15), key='-TEST_LIST-')],
            [sg.Button('Start Protocol on Selected Devices', key='-START_PROTOCOL-')]
        ]

        protocol_layout = [
             [sg.Column(protocol_editor_layout), sg.VerticalSeparator(), sg.Column(protocol_viewer_layout)]
        ]


        # Tab Group Layout
        tab_group_layout = [[sg.TabGroup([
            [sg.Tab('Grid View', grid_tab_layout, key='-GRID_TAB-'),
             sg.Tab('Individual Measurement', ind_meas_layout, key='-IND_MEAS_TAB-'),
             sg.Tab('Protocol Builder', protocol_layout, key='-PROTOCOL_TAB-')]
        ], key='-TABGROUP-')]]

        return sg.Window("Crossbar Multiplexer Control App", tab_group_layout, finalize=True) # Finalize here

    # --- Grid Helper Functions (modified update_grid_color slightly) ---
    def update_grid_color(self, row, col, selected):
        """Updates the color of a grid cell by redrawing the rectangle and updates selected count."""
        graph = self.window["-GRID-"]
        rect_id_to_delete = self.grid_rects.get((row, col)) # Use get to avoid error if ID doesn't exist
        if rect_id_to_delete:
            try:
                graph.delete_figure(rect_id_to_delete)  # Delete the old rectangle
            except Exception as e:
                 print(f"Minor Error Deleting Grid Rectangle: {e}") # Log minor error, continue

        fill_color = 'yellow' if selected else 'white'
        rect_id = graph.draw_rectangle((col, row), (col + 1, row + 1), line_color='black', fill_color=fill_color)  # Redraw with new color
        self.grid_rects[(row, col)] = rect_id  # Update rect_id in the dictionary

        # Update selected device count based on state change
        current_state = self.selection_state.get((row, col), False)
        if selected and not current_state: # Selecting a cell that was not previously selected
            self.selected_device_count += 1
        elif not selected and current_state:  # Deselecting a cell that was previously selected
            self.selected_device_count -= 1

        self.selection_state[(row, col)] = selected  # Update selection state
        self.window["-SELECTED_COUNT-"].update(f"Devices Selected: {self.selected_device_count}")  # Update count display

    def get_pin_combination_from_grid(self, row, col):
        """Map grid row and column to channel 1 and channel 2 pin numbers (1-based)."""
        ch1_pin = row + 1  # Map row (0-15) to Channel 1 pin (1-16)
        ch2_pin = col + 1  # Map column (0-15) to Channel 2 pin (1-16)
        return ch1_pin, ch2_pin

    def get_selected_devices(self):
        """Returns a list of (ch1, ch2) tuples for selected grid cells."""
        devices_to_measure = []
        for row in range(16):
            for col in range(16):
                if self.selection_state.get((row, col), False): # Check state using get
                    devices_to_measure.append(self.get_pin_combination_from_grid(row, col))
        return devices_to_measure

    # --- Backend Interaction Functions ---
    def _connect(self):
        print("Connecting equipments...")
        self.crossbar_methods.connect_multiplexer()
        self.crossbar_methods.connect_SMU()
        if self.crossbar_methods.is_mux_connected and self.crossbar_methods.is_smu_connected:
             print("All equipments connected.")
        else:
             print("One or more equipments failed to connect.")

    def _disconnect(self):
        print("Disconnecting equipments...")
        self.crossbar_methods.disconnect_multiplexer()
        self.crossbar_methods.disconnect_SMU() # Optional, handled by backend __del__ or PyVISA
        print("Equipments disconnected.")

    def _set_channels(self, ch1_str, ch2_str):
        try:
            ch1 = int(ch1_str)
            ch2 = int(ch2_str)
            if not 1 <= ch1 <= 16 or not 1 <= ch2 <= 16:
                print("Error: Channel pins must be between 1 and 16.")
                return
            print(f"Setting manual channels to CH1={ch1}, CH2={ch2}")
            self.crossbar_methods.switch_channels(ch1, ch2) # Call backend method

        except ValueError:
            print("Error: Please enter valid integer pin numbers.")
        except Exception as e:
            print(f"Error setting channels: {e}")

    def _run_single_dciv(self, values):
        try:
            ch1 = int(values["-CH1-"])
            ch2 = int(values["-CH2-"])
            pos_comp = float(values["-DCIV_POS_COMP-"])
            neg_comp = float(values["-DCIV_NEG_COMP-"])
            sweep_path = values["-SWEEP_PATH-"]
            sweep_delay_str = values["-DCIV_SWEEP_DELAY-"]
            acq_delay_str = values["-DCIV_ACQ_DELAY-"]
            plot = values["-DCIV_PLOT-"]
            save = values["-DCIV_SAVE-"]
            save_dir = values["-SAVE_FOLDER-"]

            # Validate inputs
            if not 1 <= ch1 <= 16 or not 1 <= ch2 <= 16:
                print("Error: Channel pins must be between 1 and 16.")
                return
            if not sweep_path or not os.path.exists(sweep_path):
                print(f"Error: Sweep file not found at '{sweep_path}'")
                return

            sweep_delay = float(sweep_delay_str) if sweep_delay_str else None
            acq_delay = float(acq_delay_str) if acq_delay_str else None

            print(f"\nRunning Single DCIV for CH1={ch1}, CH2={ch2}")
            self.crossbar_methods.run_single_DCIV(
                ch1, ch2, pos_comp, neg_comp, sweep_path,
                sweep_delay, acq_delay, plot, save, save_dir
            )
            print("Single DCIV finished.")

        except ValueError as e:
             print(f"Error: Invalid numeric input for DCIV parameters - {e}")
        except Exception as e:
            print(f"Error during single DCIV run: {e}")

    def _run_single_pulse(self, values):
        try:
            ch1 = int(values["-CH1-"])
            ch2 = int(values["-CH2-"])
            compliance = float(values["-PULSE_COMP-"])
            pulse_path = values["-PULSE_PATH-"]
            pulse_width_str = values["-PULSE_WIDTH-"]
            plot = values["-PULSE_PLOT-"]
            save = values["-PULSE_SAVE-"]
            save_dir = values["-SAVE_FOLDER-"]

             # Validate inputs
            if not 1 <= ch1 <= 16 or not 1 <= ch2 <= 16:
                print("Error: Channel pins must be between 1 and 16.")
                return
            if not pulse_path or not os.path.exists(pulse_path):
                print(f"Error: Pulse file not found at '{pulse_path}'")
                return

            pulse_width = float(pulse_width_str) if pulse_width_str else None

            print(f"\nRunning Single Pulse for CH1={ch1}, CH2={ch2}")
            self.crossbar_methods.run_single_pulse(
                ch1, ch2, compliance, pulse_path,
                pulse_width, plot, save, save_dir
            )
            print("Single Pulse finished.")

        except ValueError as e:
             print(f"Error: Invalid numeric input for Pulse parameters - {e}")
        except Exception as e:
            print(f"Error during single pulse run: {e}")

    def _measure_selected(self, values):
        selected_devices = self.get_selected_devices()
        if not selected_devices:
            print("No devices selected in the grid.")
            return

        save_dir = values["-SAVE_FOLDER-"]
        run_dciv = values["-GRID_DCIV-"] # Check which radio button is selected

        print(f"\nStarting measurement for {len(selected_devices)} selected devices...")

        # --- Prepare protocol based on selection ---
        protocol = []
        if run_dciv:
            try:
                 # Get DCIV params from the Individual Measurement Tab
                 dciv_params = {
                     'type': 'DCIV',
                     'params': {
                         'sweep_path': values['-SWEEP_PATH-'],
                         'pos_compl': float(values['-DCIV_POS_COMP-']),
                         'neg_compl': float(values['-DCIV_NEG_COMP-']),
                         'sweep_delay': float(values['-DCIV_SWEEP_DELAY-']) if values['-DCIV_SWEEP_DELAY-'] else None,
                         'acq_delay': float(values['-DCIV_ACQ_DELAY-']) if values['-DCIV_ACQ_DELAY-'] else None,
                         'plot': False, # Usually disable plotting for batch runs
                         'save': values["-DCIV_SAVE-"] # Use save checkbox
                     }
                 }
                 if not dciv_params['params']['sweep_path'] or not os.path.exists(dciv_params['params']['sweep_path']):
                     print(f"Error: Sweep file for grid DCIV not found at '{dciv_params['params']['sweep_path']}'. Aborting.")
                     return
                 protocol.append(dciv_params)
            except ValueError as e:
                 print(f"Error: Invalid numeric input for DCIV parameters in Individual Tab - {e}. Aborting grid measurement.")
                 return
            except Exception as e:
                 print(f"Error preparing DCIV parameters: {e}. Aborting grid measurement.")
                 return

        else: # Run Pulse
             try:
                # Get Pulse params from the Individual Measurement Tab
                 pulse_params = {
                      'type': 'PULSE',
                      'params': {
                          'pulse_path': values['-PULSE_PATH-'],
                          'compliance': float(values['-PULSE_COMP-']),
                          'pulse_width': float(values['-PULSE_WIDTH-']) if values['-PULSE_WIDTH-'] else None,
                          'plot': False, # Usually disable plotting for batch runs
                          'save': values["-PULSE_SAVE-"] # Use save checkbox
                      }
                 }
                 if not pulse_params['params']['pulse_path'] or not os.path.exists(pulse_params['params']['pulse_path']):
                     print(f"Error: Pulse file for grid Pulse not found at '{pulse_params['params']['pulse_path']}'. Aborting.")
                     return
                 protocol.append(pulse_params)
             except ValueError as e:
                 print(f"Error: Invalid numeric input for Pulse parameters in Individual Tab - {e}. Aborting grid measurement.")
                 return
             except Exception as e:
                  print(f"Error preparing Pulse parameters: {e}. Aborting grid measurement.")
                  return

        # --- Run the protocol ---
        if protocol:
            self.crossbar_methods.measure_selected(selected_devices, protocol, save_dir)
        else:
            print("Error: Could not prepare measurement protocol.")

        print("Finished measuring selected devices.")


    # --- Protocol Functions ---
    def open_test_configuration_popup(self, test_type):
        """Opens a popup to configure parameters for a specific test type."""
        # Get defaults from the main window
        sweep_path = self.window['-SWEEP_PATH-'].get()
        pulse_path = self.window['-PULSE_PATH-'].get()
        save_folder = self.window['-SAVE_FOLDER-'].get()

        if test_type == "DCIV Sweep":
            layout = [
                [sg.Text('Configure DCIV Sweep Step', font='_ 11')],
                [sg.Text('Sweep Path', size=(18,1)), sg.InputText(sweep_path, key='-SWEEP_PATH-', size=(40,1)), sg.FileBrowse()],
                [sg.Text('Positive Comp (A)', size=(18,1)), sg.InputText('0.001', key='-DCIV_POS_COMP-', size=(10,1))],
                [sg.Text('Negative Comp (A)', size=(18,1)), sg.InputText('0.01', key='-DCIV_NEG_COMP-', size=(10,1))],
                [sg.Text('Sweep Delay (s, opt)', size=(18,1)), sg.InputText('0.0001', key='-DCIV_SWEEP_DELAY-', size=(10,1))],
                 [sg.Text('Acq. Delay (s, opt)', size=(18,1)), sg.InputText('', key='-DCIV_ACQ_DELAY-', size=(10,1))],
                # No plot/save checkboxes here, controlled during execution
                [sg.Submit(), sg.Cancel()]
            ]
        elif test_type == "Pulsed Measurement":
            layout = [
                [sg.Text('Configure Pulsed Measurement Step', font='_ 11')],
                [sg.Text('Pulse Path', size=(18,1)), sg.InputText(pulse_path, key='-PULSE_PATH-', size=(40,1)), sg.FileBrowse()],
                [sg.Text('Compliance (A)', size=(18,1)), sg.InputText('0.01', key='-PULSE_COMP-', size=(10,1))],
                [sg.Text('Pulse Width (s, opt)', size=(18,1)), sg.InputText('', key='-PULSE_WIDTH-', size=(10,1))],
                # No plot/save checkboxes here
                [sg.Submit(), sg.Cancel()]
            ]
        elif test_type == "Optimize Pulse":
             layout = [
                 [sg.Text('Configure Pulse Optimization Step', font='_ 11')],
                 [sg.Text('Compliance (A)', size=(18,1)), sg.InputText('0.01', key='-PULSE_COMP-', size=(10,1))],
                 [sg.Text('Initial Vset', size=(18,1)), sg.InputText('1.0', key='-INIT_VSET-', size=(10,1))],
                 [sg.Text('Initial Vreset', size=(18,1)), sg.InputText('-1.0', key='-INIT_VRESET-', size=(10,1))],
                 [sg.Text('Initial Vread', size=(18,1)), sg.InputText('0.1', key='-INIT_VREAD-', size=(10,1))],
                 [sg.Text('Initial Width (s)', size=(18,1)), sg.InputText('1e-4', key='-INIT_WIDTH-', size=(10,1))],
                 [sg.Submit(), sg.Cancel()]
             ]
        else:
            sg.popup_error(f"Configuration popup not defined for test type: {test_type}")
            return None

        popup_window = sg.Window(f'{test_type} Config', layout)
        event, values = popup_window.read()
        popup_window.close()

        if event == 'Submit':
            # Create the config dictionary expected by the backend
            config = {'type': test_type.split()[0].upper()} # DCIV, PULSE, OPTIMIZE_PULSE etc.
            params = {}
            if test_type == "DCIV Sweep":
                params['sweep_path'] = values['-SWEEP_PATH-']
                params['pos_compl'] = float(values['-DCIV_POS_COMP-'])
                params['neg_compl'] = float(values['-DCIV_NEG_COMP-'])
                params['sweep_delay'] = float(values['-DCIV_SWEEP_DELAY-']) if values['-DCIV_SWEEP_DELAY-'] else None
                params['acq_delay'] = float(values['-DCIV_ACQ_DELAY-']) if values['-DCIV_ACQ_DELAY-'] else None
            elif test_type == "Pulsed Measurement":
                 params['pulse_path'] = values['-PULSE_PATH-']
                 params['compliance'] = float(values['-PULSE_COMP-'])
                 params['pulse_width'] = float(values['-PULSE_WIDTH-']) if values['-PULSE_WIDTH-'] else None
            elif test_type == "Optimize Pulse":
                 params['compliance'] = float(values['-PULSE_COMP-'])
                 params['initial_params'] = [
                      float(values['-INIT_VSET-']),
                      float(values['-INIT_VRESET-']),
                      float(values['-INIT_VREAD-']),
                      float(values['-INIT_WIDTH-'])
                 ]

            config['params'] = params
            return config
        return None


    def add_test_to_protocol(self, test_config):
        """Adds a test configuration to the protocol list."""
        if not test_config:
            return

        # Create a user-friendly display string
        display_string = f"{test_config['type']}: "
        params = test_config.get('params', {})
        if test_config['type'] == 'DCIV':
            path = os.path.basename(params.get('sweep_path', 'N/A'))
            display_string += f"Path={path}, PosComp={params.get('pos_compl', 'N/A')}, NegComp={params.get('neg_compl', 'N/A')}"
        elif test_config['type'] == 'PULSE':
            path = os.path.basename(params.get('pulse_path', 'N/A'))
            display_string += f"Path={path}, Comp={params.get('compliance', 'N/A')}, Width={params.get('pulse_width', 'Default')}"
        elif test_config['type'] == 'OPTIMIZE_PULSE':
             init = params.get('initial_params', [])
             display_string += f"Comp={params.get('compliance', 'N/A')}, Init=[{','.join(map(str, init))}]"
        else:
            display_string += str(params) # Generic display for other types

        self.protocol_list_configs.append(test_config) # Store the full config
        current_display_list = self.window['-TEST_LIST-'].get_list_values()
        current_display_list.append(display_string) # Add display string to Listbox
        self.window['-TEST_LIST-'].update(values=current_display_list)

    def remove_selected_test(self):
        """Removes the selected test from the protocol."""
        selected_indices = self.window['-TEST_LIST-'].get_indexes()
        if not selected_indices:
            print("No test selected to remove.")
            return

        selected_index = selected_indices[0] # Get the index of the selected item

        # Remove from both the display list and the config list
        current_display_list = list(self.window['-TEST_LIST-'].get_list_values())
        del current_display_list[selected_index]
        del self.protocol_list_configs[selected_index]

        self.window['-TEST_LIST-'].update(values=current_display_list)
        print("Selected test removed from protocol.")

    def clear_protocol(self):
         """Clears all tests from the protocol."""
         self.protocol_list_configs = []
         self.window['-TEST_LIST-'].update(values=[])
         print("Protocol cleared.")

    def execute_protocol(self, values):
        """Executes the defined protocol on selected devices."""
        selected_devices = self.get_selected_devices()
        if not selected_devices:
            print("No devices selected in the grid.")
            return

        if not self.protocol_list_configs:
             print("Protocol is empty. Add tests first.")
             return

        save_dir = values["-SAVE_FOLDER-"]
        print(f"\nStarting protocol execution for {len(selected_devices)} devices...")
        print(f"Protocol Steps: {len(self.protocol_list_configs)}")

        # Call the backend method to handle the execution
        self.crossbar_methods.measure_selected(selected_devices, self.protocol_list_configs, save_dir)

        print("Protocol execution finished.")


    # --- Main Loop ---
    def run(self):
        #self.window.Finalize() # Already finalized in create_window

        # Draw Grid and initialize selection state
        graph = self.window["-GRID-"]
        for row in range(16):
            for col in range(16):
                rect_id = graph.draw_rectangle((col, row), (col + 1, row + 1), line_color='black', fill_color='white')
                self.grid_rects[(row, col)] = rect_id
                self.selection_state[(row, col)] = False

        # Add pin numbers to the grid
        for i in range(1, 17):
            graph.draw_text(text=str(i), location=(i-0.5, 16.5), color='black', font='_ 8') # Top numbers (Ch2)
            graph.draw_text(text=str(i), location=(-0.5, i-0.5), color='black', font='_ 8') # Side numbers (Ch1)

        # --- Event Loop ---
        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                self._disconnect() # Ensure disconnection on exit
                break

            # --- Connection ---
            elif event == "-CONNECT-":
                self._connect()
            elif event == "-DISCONNECT-":
                self._disconnect()

            # --- Grid Events ---
            elif event == "-GRID-":
                click_pos = values["-GRID-"]
                if click_pos:
                    col, row = int(click_pos[0]), int(click_pos[1])
                    if 0 <= row < 16 and 0 <= col < 16:
                        self.update_grid_color(row, col, not self.selection_state.get((row, col), False)) # Toggle

            elif event == "-SELECT_ALL-":
                for r in range(16):
                    for c in range(16):
                        if not self.selection_state.get((r, c), False): # Only update if not already selected
                            self.update_grid_color(r, c, True)

            elif event == "-DESELECT_ALL-":
                for r in range(16):
                    for c in range(16):
                         if self.selection_state.get((r, c), False): # Only update if currently selected
                             self.update_grid_color(r, c, False)

            elif event == "-SELECT_HALF-":
                for r in range(16):
                    for c in range(16):
                        is_selected = (r + c) % 2 == 0
                        self.update_grid_color(r, c, is_selected)

            elif event == "-TOGGLE_SELECTION-":
                 for r in range(16):
                    for c in range(16):
                        self.update_grid_color(r, c, not self.selection_state.get((r,c), False))

            elif event == "-MEASURE_SELECTED-":
                 self._measure_selected(values)

            # --- Individual Measurement Events ---
            elif event == "-SET_CHANNELS-":
                 self._set_channels(values["-CH1-"], values["-CH2-"])
            elif event == "-RUN_SINGLE_DCIV-":
                self._run_single_dciv(values)
            elif event == "-RUN_SINGLE_PULSE-":
                 self._run_single_pulse(values)

            # --- Protocol Events ---
            elif event == '-ADD_TEST-':
                 selected_test_type = values['-TEST_TYPE-']
                 if selected_test_type:
                      test_config = self.open_test_configuration_popup(selected_test_type)
                      self.add_test_to_protocol(test_config)
                 else:
                      print("Please select a test type to add.")

            elif event == '-REMOVE_TEST-':
                 self.remove_selected_test()
            elif event == '-CLEAR_PROTOCOL-':
                 self.clear_protocol()
            elif event == '-START_PROTOCOL-':
                 self.execute_protocol(values)


        self.window.close()


if __name__ == "__main__":
    gui = CrossbarGUI()
    gui.run()