# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 11:43:55 2025
@author:Shrey


ViPSA Advanced GUI based on Viewfinder3.py and Main4.py
Includes a Protocol Builder tab.
"""

import os
import PySimpleGUI as sg
import cv2
import threading
import time
import pandas as pd
import csv

# Import ViPSA backend and helper classes
from Main4 import Vipsa_Methods
from Vision import overlay
from Openflexture import Zaber
from ProtocolEditor import ProtocolBuilder, ProtocolStepEditor

# --- Default Paths and Settings ---
DEFAULT_SAVE_DIRECTORY = "C:/Users/amdm/OneDrive - Nanyang Technological University/ViPSA data folder/"
DEFAULT_SWEEP_PATH = "C:/Users/amdm/OneDrive - Nanyang Technological University/ViPSA data folder/sweep patterns"
DEFAULT_PULSE_PATH = "C:/Users/amdm/Desktop/sweep patterns/Pulse20.csv"

class VipsaGUI:
    def __init__(self):
        sg.theme('DarkBlue')
        self.vipsa = Vipsa_Methods()
        self.protocol_builder = ProtocolBuilder(None, self.vipsa)
        self.window = self.create_window()

        # State variables
        self.camera_thread = None
        self.camera_running = False
        self._cap = None
        
        self.is_equipment_connected = False

    def create_window(self):
        # =================== Tab Layouts ===================

        # --- Movement Tab ---
        # (Adapted from Viewfinder3.py)
        movement_tab = [
             [sg.Column([
                 [sg.Text("Camera Feed")],
                 [sg.Image(filename='', key='-CAMERA_FEED-', size=(640, 480))],
                 [sg.Button("Start Camera", key='-START_CAMERA-'), sg.Button("Stop Camera", key='-STOP_CAMERA-')],
             ], vertical_alignment='top'),
             sg.VerticalSeparator(),
             sg.Column([
                 [sg.Text("--- Connections ---", font='_ 11')],
                 [sg.Button("Connect All Equipments", key='-CONNECT_ALL-'), sg.Button("Disconnect All", key='-DISCONNECT_ALL-')],
                 [sg.Text("Status: Disconnected", key='-CONNECTION_STATUS-', text_color='red')],
                 [sg.HSeparator()],
                 [sg.Text("--- Arduino Stage (OpenFlexure) ---", font='_ 11')],
                 [sg.Text("Port:", size=(8,1)), sg.InputText(default_text='COM5', key='-ARDUINO_PORT-', size=(10, 1)),
                  sg.Text("Baud:", size=(5,1)), sg.InputText(default_text='115200', key='-ARDUINO_BAUD-', size=(10, 1)),
                  sg.Text("Scale:", size=(5,1)), sg.InputText(default_text='1', key='-ARDUINO_SCALE-', size=(10, 1))],
                 [sg.Text("X Steps:", size=(8,1)), sg.InputText(default_text='10', key='-X_STEPS-', size=(5, 1)), sg.Button("X+", key='-MOVE_X_POS-', size=(3,1)), sg.Button("X-", key='-MOVE_X_NEG-', size=(3,1))],
                 [sg.Text("Y Steps:", size=(8,1)), sg.InputText(default_text='10', key='-Y_STEPS-', size=(5, 1)), sg.Button("Y+", key='-MOVE_Y_POS-', size=(3,1)), sg.Button("Y-", key='-MOVE_Y_NEG-', size=(3,1))],
                 [sg.Text("Z Steps:", size=(8,1)), sg.InputText(default_text='10', key='-Z_STEPS-', size=(5, 1)), sg.Button("Z+", key='-MOVE_Z_POS-', size=(3,1)), sg.Button("Z-", key='-MOVE_Z_NEG-', size=(3,1))],
                 [sg.Button("Set Zero", key='-SET_ZERO-', size=(10,1)), sg.Button("Go to Zero", key='-GO_ZERO-', size=(10,1))],
                 [sg.HSeparator()],
                 [sg.Text("--- Zaber Stage ---", font='_ 11')],
                 [sg.Text("Port:", size=(8,1)), sg.InputText(default_text='COM7', key='-ZABER_PORT-', size=(10, 1))],
                 [sg.Text("X Dist:", size=(8,1)), sg.InputText(default_text='1000', key='-ZABER_X_DIST-', size=(8, 1)), sg.Button("X+", key='-MOVE_ZABER_X_POS-', size=(3,1)), sg.Button("X-", key='-MOVE_ZABER_X_NEG-', size=(3,1))],
                 [sg.Text("Y Dist:", size=(8,1)), sg.InputText(default_text='1000', key='-ZABER_Y_DIST-', size=(8, 1)), sg.Button("Y+", key='-MOVE_ZABER_Y_POS-', size=(3,1)), sg.Button("Y-", key='-MOVE_ZABER_Y_NEG-', size=(3,1))],
                 [sg.HSeparator()],
                 [sg.Text("--- Lights ---", font='_ 11')],
                 [sg.Button("Turn On Lights", key='-TURN_ON_LIGHTS-'), sg.Button("Turn Off Lights", key='-TURN_OFF_LIGHTS-')],
                 [sg.HSeparator()],
                 [sg.Text("--- SMU ---", font='_ 11')],
                 [sg.Text("Select SMU:", size=(10,1)),
                    sg.Combo(
                        values=["Keithley2450", "KeysightB2901BL"],
                        default_value="Keithley2450",
                        key="-SMU_SELECT-",
                        readonly=True,
                        size=(20,1))],
                 [sg.Button("Quick Align ⌖", key = '-QUICK_ALIGN-'), sg.Button("Quick Approach ☟", key = '-QUICK_APPROACH-' )]
             ], vertical_alignment='top')]
        ]

        # --- Grid Tab ---
        # (Adapted from Viewfinder3.py, potential for grid selection later)
        grid_tab = [
             [sg.Column([
                 [sg.Text("Camera Feed")],
             [sg.Image(filename='', key='-GRID_CAMERA_FEED-', size=(640, 480))],
             [sg.Button("Start Camera", key='-GRID_START_CAMERA-'), sg.Button("Stop Camera", key='-GRID_STOP_CAMERA-')],
             ], vertical_alignment='top'),
             sg.VerticalSeparator(),
             sg.Column([
                 [sg.Text("--- Grid Creation ---", font='_ 11')],
                 [sg.Text("X Distance:", size=(10,1)), sg.InputText(key='-X_DIST-', size=(10, 1))],
                 [sg.Text("Y Distance:", size=(10,1)), sg.InputText(key='-Y_DIST-', size=(10, 1))],
                 [sg.Text("Rows:", size=(10,1)), sg.InputText(key='-ROWS-', size=(10, 1))],
                 [sg.Text("Columns:", size=(10,1)), sg.InputText(key='-COLS-', size=(10, 1))],
                 [sg.Text("Grid Save Path:", size=(15,1)), sg.InputText(key='-GRID_PATH-', size=(40, 1)), sg.FolderBrowse()],
                 [sg.Button("Create Grid CSV", key='-CREATE_GRID-')],
                 [sg.HSeparator()],
                 [sg.Text("--- Grid Measurement ---", font='_ 11')],
                 [sg.Text("Grid File:", size=(15,1)), sg.InputText(key='-GRID_MEAS_PATH-', size=(40, 1)), sg.FileBrowse()],
                 [sg.Text("Sample ID:", size=(15,1)), sg.InputText("sample", key='-GRID_SAMPLE_ID-', size=(15,1))],
                 [sg.Text("Start Device #:", size=(15,1)), sg.InputText("1", key='-GRID_STARTPOINT-', size=(5, 1))],
                 [sg.Text("Skip Devices:", size=(15,1)), sg.InputText("1", key='-GRID_SKIP-', size=(5, 1))],
                 [sg.Checkbox("Randomize Order", key='-GRID_RANDOMIZE-')],
                 [sg.Text("Use Measurement Protocol Below:")],
                 [sg.Button("Run Grid Measurement (DCIV)", key='-RUN_GRID_DCIV-')],
                 [sg.Button("Run Grid Measurement (Pulse)", key='-RUN_GRID_PULSE-')],
                 [sg.Button("Run Grid Measurement (Protocol)", key='-RUN_GRID_PROTOCOL-')], # Added button for protocol on grid
                 [sg.HSeparator()],
                 [sg.Text("Parameters (Used for Grid DCIV/Pulse):", font='_ 10')],
                 [sg.Text("Pos Comp:", size=(10,1)), sg.InputText('0.001', key='-GRID_POS_COMP-', size=(10,1)),
                  sg.Text("Neg Comp:", size=(10,1)), sg.InputText('0.01', key='-GRID_NEG_COMP-', size=(10,1))],
                 [sg.Text("Pulse Comp:", size=(10,1)), sg.InputText('0.01', key='-GRID_PULSE_COMP-', size=(10,1))],
                 [sg.Text("Sweep Path:", size=(10,1)), sg.InputText(DEFAULT_SWEEP_PATH, key='-GRID_SWEEP_PATH-', size=(40,1)), sg.FileBrowse()],
                 [sg.Text("Pulse Path:", size=(10,1)), sg.InputText(DEFAULT_PULSE_PATH, key='-GRID_PULSE_PATH-', size=(40,1)), sg.FileBrowse()],
                  [sg.Text("Sweep Delay:", size=(10,1)), sg.InputText('0.0001', key='-GRID_SWEEP_DELAY-', size=(10,1))],
                   [sg.Text("Pulse Width:", size=(10,1)), sg.InputText('0.001', key='-GRID_PULSE_WIDTH-', size=(10,1))],
             ], vertical_alignment='top')]
        ]


        # --- Measure Tab ---
        # (Adapted from Viewfinder3.py for single measurements)
        measure_tab = [
            [sg.Text("--- Single Measurement Target ---", font='_ 11')],
            [sg.Text("Sample ID:", size=(10,1)), sg.InputText("sample", key='-MEAS_SAMPLE_ID-', size=(15, 1)),
             sg.Text("Device #:", size=(8,1)), sg.InputText("0", key='-MEAS_DEVICE_ID-', size=(5, 1))],
            [sg.Text("Save Folder:", size=(10,1)), sg.InputText(DEFAULT_SAVE_DIRECTORY, key='-MEAS_SAVE_FOLDER-', size=(50, 1)), sg.FolderBrowse()],
             [sg.Checkbox("Align First", default=False, key='-MEAS_ALIGN-'),
              sg.Checkbox("Approach First", default=False, key='-MEAS_APPROACH-'),
              sg.Text("Wait betⁿ measurements for:", size=(10,1)), sg.InputText("0", key='-WAIT_TIME-', size=(5, 1))],
            [sg.Button("Measure resistance", key='-RUN_MEAS_RESISTANCE-'), sg.Text("Checking Voltage (V):",size=(10,1)), sg.InputText("0.01", key ='-MEAS_RES_VOLTAGE-', size=(10,1))],

            [sg.HSeparator()],
            [sg.Text("--- DCIV Sweep Parameters ---", font='_ 11')],
            [sg.Text("Sweep Path:", size=(15,1)), sg.InputText(DEFAULT_SWEEP_PATH, key='-MEAS_SWEEP_PATH-', size=(50, 1)), sg.FileBrowse()],
            [sg.Text("Pos Comp (A):", size=(15,1)), sg.InputText(default_text='0.001', key='-MEAS_POS_COMP-', size=(10, 1)),
             sg.Text("Neg Comp (A):", size=(15,1)), sg.InputText(default_text='0.01', key='-MEAS_NEG_COMP-', size=(10, 1))],
            [sg.Text("Sweep Delay (s):", size=(15,1)), sg.InputText(default_text='0.0001', key='-MEAS_SWEEP_DELAY-', size=(10,1))],
            [sg.Checkbox('Plot Result', default=True, key='-MEAS_DCIV_PLOT-')],
            [sg.Button("Run Single DCIV Measurement", key='-RUN_SINGLE_DCIV_MEAS-')],

            [sg.HSeparator()],
            [sg.Text("--- Pulsed Measurement Parameters ---", font='_ 11')],
            [sg.Text("Pulse Path:", size=(15,1)), sg.InputText(DEFAULT_PULSE_PATH, key='-MEAS_PULSE_PATH-', size=(50, 1)), sg.FileBrowse()],
            [sg.Text("Compliance (A):", size=(15,1)), sg.InputText(default_text='0.01', key='-MEAS_PULSE_COMP-', size=(10, 1))],
            [sg.Text("Pulse Width (s):", size=(15,1)), sg.InputText(default_text='0.001', key='-MEAS_PULSE_WIDTH-', size=(10,1))],
            [sg.Checkbox('Plot Result', default=True, key='-MEAS_PULSE_PLOT-')],
             [sg.Button("Run Single Pulsed Measurement", key='-RUN_SINGLE_PULSE_MEAS-')]
        ]

        # --- Protocol Builder Tab ---
        protocol_editor_layout = [
             [sg.Text('Protocol Builder', font='_ 12 bold')],
             [sg.Text('Build complex measurement sequences with full parameter control.', text_color='lightblue')],
             [sg.Button('➕ Add Step', key='-ADD_TEST-', size=(15, 1)), 
              sg.Button('✏ Edit Selected', key='-EDIT_TEST-', size=(15, 1)),
              sg.Button('🗑 Remove Selected', key='-REMOVE_TEST-', size=(15, 1)),
              sg.Button('Clear All', key='-CLEAR_PROTOCOL-', size=(12, 1))],
             [sg.Button('💾 Save Protocol', key='-SAVE_PROTOCOL-', size=(15, 1)), 
              sg.Button('📂 Load Protocol', key='-LOAD_PROTOCOL-', size=(15, 1))],
        ]
        protocol_viewer_layout = [
             [sg.Text('Protocol Steps:', font='_ 11 bold')],
             [sg.Listbox(values=[], size=(60, 12), key='-TEST_LIST-', enable_events=True)],
             [sg.Text('Double-click to edit • Right-click to remove', text_color='gray', size=(40, 1))],
             [sg.Button('▶ Run Protocol on Current Target', key='-RUN_PROTOCOL_SINGLE-', size=(38, 1), button_color=('white', 'green'))],
        ]

        protocol_tab = [[sg.Column(protocol_editor_layout), sg.VerticalSeparator(), sg.Column(protocol_viewer_layout)]]

        # --- Output Log Layout ---
        output_layout = [
             [sg.Text("--- Log Output ---", font='_ 9')],
             # Make the Multiline element expand horizontally and vertically
             [sg.Multiline(size=(120, 50),
                           key="-OUTPUT-",
                           autoscroll=True,
                           disabled=True,
                           reroute_stdout=True,
                           reroute_stderr=True,
                           expand_x=True,
                           expand_y=False)]
        ]

        # --- Refactored Main Layout: TabGroup on the left, Output Log on the right ---

        # Define the TabGroup *without* the Output Log
        tab_group = sg.TabGroup(
            [[
                sg.Tab('Movement & Connections', movement_tab, key='-TAB_MOVE-'),
                sg.Tab('Grid Creation & Measurement', grid_tab, key='-TAB_GRID-'),
                sg.Tab('Single Measurement', measure_tab, key='-TAB_MEASURE-'),
                sg.Tab('Protocol Builder', protocol_tab, key='-TAB_PROTOCOL-'),
            ]],
            expand_x=True,
            expand_y=True
        )
        
        # Put your output label + multiline into a right-side column
        output_column = sg.Column(
            [
                output_layout[0],
                output_layout[1],
            ],
            expand_y=True,
            vertical_alignment='top'
        )
        
        layout = [[tab_group, sg.VerticalSeparator(), output_column]]


        return sg.Window("ViPSA Control Center", layout, finalize=True, resizable=True)

    # =================== Event Handlers ===================

    # --- Camera Handling ---
    def _start_camera_thread(self):
        """Starts the camera feed in a separate thread."""
        if self.camera_thread is None or not self.camera_thread.is_alive():
            self.camera_running = True
            # ensure previous capture ref cleared
            self._cap = None
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
            print("Camera thread started.")
        else:
             print("Camera thread already running.")

    def _stop_camera_thread(self):
        """Signals the camera thread to stop."""
        if self.camera_thread and self.camera_thread.is_alive():
            # Signal the loop to stop
            self.camera_running = False
            # Release the capture to unblock any blocking read() call
            try:
                if getattr(self, '_cap', None) is not None:
                    try:
                        if self._cap.isOpened():
                            self._cap.release()
                    except Exception:
                        pass
            except Exception:
                pass

            # Wait for the thread to finish
            self.camera_thread.join(timeout=2.0)
            if self.camera_thread.is_alive():
                print("Warning: Camera thread did not stop gracefully.")
            else:
                print("Camera thread stopped.")
            self.camera_thread = None
            # Optionally clear the image widgets
            try:
                self.window['-CAMERA_FEED-'].update(data=None)
                self.window['-GRID_CAMERA_FEED-'].update(data=None)
            except Exception:
                pass

        else:
            print("Camera thread not running.")


    def _camera_loop(self):
        """The main loop for capturing and displaying camera frames."""
        print("Camera loop starting...")
        try:
             cap = cv2.VideoCapture(0) # Use camera index 0
             # publish capture so main thread can release it when stopping
             try:
                 self._cap = cap
             except Exception:
                 pass

             if not cap.isOpened():
                 print("Error: Could not open camera.")
                 self.camera_running = False
                 return

             while self.camera_running:
                 ret, frame = cap.read()
                 if not ret:
                     # If capture was released, break out
                     if getattr(self, '_cap', None) is None or (hasattr(self._cap, 'isOpened') and not self._cap.isOpened()):
                         break
                    
                     print("Error: Could not read frame.")
                     time.sleep(0.1) # Avoid busy-waiting on error
                     continue

                 try:
                     # Process frame (apply overlay)
                     processed_frame = overlay(frame) # From Vision.py
                     # Encode frame for PySimpleGUI
                     imgbytes = cv2.imencode('.png', processed_frame)[1].tobytes()
                     # Update both camera widgets - use write_event_value for thread safety
                     try:
                         if getattr(self, 'window', None) is not None:
                             self.window.write_event_value('-UPDATE_CAMERA_FEED-', imgbytes)
                     except Exception as e:
                         # Window might be closed; print and exit loop
                         print(f"Camera thread: failed to send frame event: {e}")
                         break

                 except Exception as e:
                      print(f"Error processing frame: {e}")


                 time.sleep(0.05)

        except Exception as e:
             print(f"Error in camera loop: {e}")
        finally:
             try:
                 if getattr(self, '_cap', None) is not None:
                     try:
                         if self._cap.isOpened():
                             self._cap.release()
                     except Exception:
                         pass
                     self._cap = None
             except Exception:
                 pass
             print("Camera loop finished.")
             self.camera_running = False # Ensure flag is reset


    # --- Connection Handling ---
    def _connect_all(self):
        print("Connecting all equipment...")
        selected_smu = self.window['-SMU_SELECT-'].get()
        print(f"Selected SMU: {selected_smu}")
        success = self.vipsa.connect_equipment(SMU_name=selected_smu)
        self.is_equipment_connected = success
        if success:
            print("All equipment connected successfully.")
            self.window['-CONNECTION_STATUS-'].update('Status: Connected', text_color='green')
        else:
            print("Failed to connect all equipment.")
            self.window['-CONNECTION_STATUS-'].update('Status: Error', text_color='red')

    def _disconnect_all(self):
        print("Disconnecting all equipment...")
        self.vipsa.disconnect_equipment() # Backend handles individual disconnections
        self.is_equipment_connected = False
        print("All equipment disconnected.")
        self.window['-CONNECTION_STATUS-'].update('Status: Disconnected', text_color='red')

    # --- Movement Handling ---
    def _handle_arduino_move(self, axis, steps_key, direction):
        if not self.is_equipment_connected or not self.vipsa.stage:
             print("Error: Arduino stage not connected.")
             return
        try:
             steps = float(self.window[steps_key].get())
             steps = steps*direction
             
             print(f"Moving Arduino {axis} by {steps} steps...")
             if axis == 'X': self.vipsa.stage.move_x_by(steps)
             elif axis == 'Y': self.vipsa.stage.move_y_by(steps)
             elif axis == 'Z': self.vipsa.stage.move_z_by(steps)
             print("Arduino move complete.")
        except ValueError:
             print("Error: Invalid step value entered.")
        except Exception as e:
             print(f"Error moving Arduino {axis}: {e}")

    def _handle_zaber_move(self, axis, dist_key, direction):
        if not self.is_equipment_connected or not self.vipsa.zaber_x or not self.vipsa.zaber_y:
             print("Error: Zaber stages not connected.")
             return
        try:
             distance = float(self.window[dist_key].get())
             distance = direction*distance
             
             print(f"Moving Zaber {axis} by {distance}...")
             if axis == 'X': self.vipsa.zaber_x.move_relative(distance)
             elif axis == 'Y': self.vipsa.zaber_y.move_relative(distance)
             print("Zaber move complete.")
        except ValueError:
             print("Error: Invalid distance value entered.")
        except Exception as e:
             print(f"Error moving Zaber {axis}: {e}")


    # --- Light Handling ---
    def _control_lights(self, state):
         if not self.is_equipment_connected or not self.vipsa.top_light:
              print("Error: Lights not connected.")
              return
         try:
              print(f"Turning lights {state}...")
              self.vipsa.top_light.control_lights(state)
              print("Light command sent.")
         except Exception as e:
              print(f"Error controlling lights: {e}")

    # --- Grid Creation ---
    def _create_grid_csv(self, values):
        try:
             x_dist = float(values['-X_DIST-'])
             y_dist = float(values['-Y_DIST-'])
             rows = int(values['-ROWS-'])
             cols = int(values['-COLS-'])
             save_path = values['-GRID_PATH-']

             if not save_path:
                  print("Error: Please select a Grid Save Path.")
                  return
             if not self.vipsa.zaber_x or not self.vipsa.zaber_y:
                 # Try to connect Zaber if not already connected - needed for start pos
                 try:
                     print("Connecting Zaber for grid creation...")
                     self.vipsa.Zaber = Zaber('COM3') # Use default or get from input
                     self.vipsa.zaber_x, self.vipsa.zaber_y = self.vipsa.Zaber.get_devices()
                     print("Zaber connected.")
                 except Exception as e:
                      print(f"Error: Zaber stages needed for grid creation but couldn't connect: {e}")
                      return


             # Get the start position from Zaber
             start_x = self.vipsa.zaber_x.get_position()
             start_y = self.vipsa.zaber_y.get_position()

             grid_points = []
             device_number = 1
             for row in range(rows):
                 for col in range(cols):
                     # Compute coordinates (simple rectangular for now)
                     x_coord = start_x + col * x_dist
                     y_coord = start_y + row * y_dist
                     grid_points.append([device_number, x_coord, y_coord])
                     device_number += 1

             # Save the grid to a CSV file
             os.makedirs(save_path, exist_ok=True) # Ensure directory exists
             filepath = os.path.join(save_path, "grid.csv")
             with open(filepath, mode='w', newline='') as file:
                  writer = csv.writer(file)
                  writer.writerow(['Device', 'X', 'Y']) # Header
                  writer.writerows(grid_points)

             print(f"Grid CSV created successfully at {filepath}")
             # Update the Grid Measurement Path input
             self.window['-GRID_MEAS_PATH-'].update(filepath)

        except ValueError:
             print("Error: Invalid numeric input for grid parameters.")
        except Exception as e:
             print(f"Error creating grid CSV: {e}")
             
    # --- Probing Utils ---
    
    def _quick_align(self):
        
        if not self.is_equipment_connected:
            print("Error: Equipment not connected.")
            return
        try:
            self.vipsa.correct_course()
            
        except Exception as e:
            print(f"Encountered an issue during align ! \n {e} \n(perhaps you left the camera on ?)")
            
    def _quick_approach(self):
        
        if not self.is_equipment_connected:
            print("Error, equipment not connected")
            return
        try:
            print("Performing Approach")
            self.vipsa.detect_contact_and_move_z()
            
        except Exception as e:
            print(f"Error during approach : {e}")
            
    
    # --- Measurement Execution ---
    
    def _run_measure_resistance(self, values):
        if not self.is_equipment_connected:
            print("Error: Equipment not connected.")
            return
        try:
            voltage = values['-MEAS_RES_VOLTAGE-']
            self.vipsa.SMU.get_contact_current(voltage)
            
        except ValueError:
            print("Error: Invalid numeric value for measurement parameters.")
        except Exception as e:
            print(f"Error during single DCIV measurement: {e}")
    
    def _run_single_dciv_meas(self, values):
        if not self.is_equipment_connected:
             print("Error: Equipment not connected.")
             return
        try:
             # Gather parameters
             sample_id = values['-MEAS_SAMPLE_ID-']
             device_id = values['-MEAS_DEVICE_ID-']
             save_dir = values['-MEAS_SAVE_FOLDER-']
             align = values['-MEAS_ALIGN-']
             approach = values['-MEAS_APPROACH-']
             sweep_path = values['-MEAS_SWEEP_PATH-']
             pos_comp = float(values['-MEAS_POS_COMP-'])
             neg_comp = float(values['-MEAS_NEG_COMP-'])
             sweep_delay_str = values['-MEAS_SWEEP_DELAY-']
             plot = values['-MEAS_DCIV_PLOT-']
             wait_time = float(values['-WAIT_TIME-'])
             
             compliance_pf = pos_comp
             compliance_pb = pos_comp
             compliance_nf = neg_comp
             compliance_nb = pos_comp

             sweep_delay = float(sweep_delay_str) if sweep_delay_str else None

             if not sweep_path or not os.path.exists(sweep_path):
                  print(f"Error: Sweep file not found: {sweep_path}")
                  return
             
             print(f"\nStarting Single DCIV for Sample: {sample_id}, Device: {device_id}")
             # Call backend function - it handles internal connections
             is_measured, height, saved_file_path = self.vipsa.run_single_DCIV(
                 sample_no=sample_id, device_no=device_id,
                 pos_compl=pos_comp, neg_compl=neg_comp, sweep_delay=sweep_delay,
                 plot=plot, align=align, approach=approach,
                 save_directory=save_dir, sweep_path=sweep_path, wait_time=wait_time,
                 compliance_pf=compliance_pf, compliance_pb=compliance_pb,
                 compliance_nf=compliance_nf, compliance_nb=compliance_nb, use_4way_split=True,
                 SMU=None, stage=None, Zaber_x=None, Zaber_y=None, top_light=None, 
             )

             if is_measured:
                  print(f"Single DCIV finished. Data saved to: {saved_file_path}")
                  print(f"Device Height after measurement: {height}")
             else:
                  print("Single DCIV failed or contact not established.")

        except ValueError:
            print("Error: Invalid numeric value for measurement parameters.")
        except Exception as e:
            print(f"Error during single DCIV measurement: {e}")

    def _run_single_pulse_meas(self, values):
        if not self.is_equipment_connected:
             print("Error: Equipment not connected.")
             return
        try:
             # Gather parameters
             sample_id = values['-MEAS_SAMPLE_ID-']
             device_id = values['-MEAS_DEVICE_ID-']
             save_dir = values['-MEAS_SAVE_FOLDER-']
             align = values['-MEAS_ALIGN-']
             approach = values['-MEAS_APPROACH-']
             pulse_path = values['-MEAS_PULSE_PATH-']
             compliance = float(values['-MEAS_PULSE_COMP-'])
             pulse_width_str = values['-MEAS_PULSE_WIDTH-']
             plot = values['-MEAS_PULSE_PLOT-']

             pulse_width = float(pulse_width_str) if pulse_width_str else None

             if not pulse_path or not os.path.exists(pulse_path):
                 print(f"Error: Pulse file not found: {pulse_path}")
                 return

             print(f"\nStarting Single Pulse for Sample: {sample_id}, Device: {device_id}")
             # Call backend function
             is_measured, height, saved_file_path = self.vipsa.run_single_pulse(
                 sample_no=sample_id, device_no=device_id,
                 compliance=compliance, pulse_width=pulse_width,
                 plot=plot, align=align, approach=approach,
                 save_directory=save_dir, pulse_path=pulse_path,
                 SMU=None, stage=None, Zaber_x=None, Zaber_y=None, top_light=None
             )

             if is_measured:
                 print(f"Single Pulse finished. Data saved to: {saved_file_path}")
                 print(f"Device Height after measurement: {height}")
             else:
                 print("Single Pulse failed or contact not established.")

        except ValueError:
            print("Error: Invalid numeric value for measurement parameters.")
        except Exception as e:
            print(f"Error during single pulse measurement: {e}")

    # --- Grid Measurement Execution ---
    def _run_grid_measurement(self, values, measurement_type='DCIV'):
         if not self.is_equipment_connected:
              print("Error: Equipment not connected.")
              return
         try:
             grid_path = values['-GRID_MEAS_PATH-']
             sample_id = values['-GRID_SAMPLE_ID-']
             startpoint = int(values['-GRID_STARTPOINT-'])
             skip = int(values['-GRID_SKIP-'])
             randomize = values['-GRID_RANDOMIZE-']
             save_dir = values['-MEAS_SAVE_FOLDER-'] # Use save folder from Measure tab for consistency

             if not grid_path or not os.path.exists(grid_path):
                  print(f"Error: Grid file not found: {grid_path}")
                  return

             print(f"\nStarting Grid Measurement ({measurement_type})...")

             if measurement_type == 'DCIV':
                  pos_comp = float(values['-GRID_POS_COMP-'])
                  neg_comp = float(values['-GRID_NEG_COMP-'])
                  sweep_delay = float(values['-GRID_SWEEP_DELAY-']) if values['-GRID_SWEEP_DELAY-'] else None
                  sweep_path = values['-GRID_SWEEP_PATH-']
                  if not sweep_path or not os.path.exists(sweep_path):
                      print(f"Error: Grid DCIV sweep file not found: {sweep_path}")
                      return

                  self.vipsa.measure_IV_gridwise(
                       sample_ID=sample_id, gridpath=grid_path,
                       pos_compl=pos_comp, neg_compl=neg_comp, sweep_delay=sweep_delay,
                       skip_instances=skip, startpoint=startpoint, randomize=randomize,
                       plot=False, # Plotting disabled for grid runs
                       align=True, approach=True, # Default align/approach for grid
                       save_directory=save_dir, sweep_path=sweep_path,
                       SMU=None, stage=None, Zaber_x=None, Zaber_y=None, top_light=None
                  )
             elif measurement_type == 'PULSE':
                   pulse_comp = float(values['-GRID_PULSE_COMP-'])
                   pulse_width = float(values['-GRID_PULSE_WIDTH-']) if values['-GRID_PULSE_WIDTH-'] else None
                   pulse_path = values['-GRID_PULSE_PATH-']
                   if not pulse_path or not os.path.exists(pulse_path):
                        print(f"Error: Grid Pulse file not found: {pulse_path}")
                        return
                   print("Running PULSE grid measurement for each device...")
                   grid_data = pd.read_csv(grid_path)
                   device_coords = grid_data[['Device', 'X', 'Y']].values.tolist()
                   for i, (dev_id, x, y) in enumerate(device_coords):
                        if i < (startpoint -1): continue # Skip until startpoint
                        if (i - (startpoint - 1)) % skip != 0: continue # Handle skip

                        print(f"\n--- Grid Pulse: Device {int(dev_id)} ---")
                        try:
                            self.vipsa.zaber_x.move_absolute(x)
                            self.vipsa.zaber_y.move_absolute(y)
                        except Exception as e:
                            print(f"Error moving Zaber to device {int(dev_id)}: {e}")
                            continue

                        try:
                            self.vipsa.run_single_pulse(
                                sample_no=sample_id, device_no=int(dev_id),
                                compliance=pulse_comp, pulse_width=pulse_width,
                                plot=False, align=True, approach=True,
                                save_directory=save_dir, pulse_path=pulse_path,
                                SMU=None, stage=None, Zaber_x=None, Zaber_y=None, top_light=None
                            )
                        except Exception as e:
                            print(f"Error running pulse on device {int(dev_id)}: {e}")
                   print("Grid Pulse Measurement loop finished.")


             elif measurement_type == 'PROTOCOL':
                  if not self.protocol_list_configs:
                       print("Error: Protocol is empty. Cannot run grid protocol.")
                       return
                  print("Executing protocol on grid devices...")

                  grid_data = pd.read_csv(grid_path)
                  device_coords = grid_data[['Device', 'X', 'Y']].values.tolist()

                  for i, (dev_id, x, y) in enumerate(device_coords):
                      if i < (startpoint -1): continue # Skip until startpoint
                      if (i - (startpoint - 1)) % skip != 0: continue # Handle skip

                      print(f"\n--- Grid Protocol: Device {int(dev_id)} ---")
                      try:
                          # Move stages to device
                          if self.vipsa.zaber_x and self.vipsa.zaber_y:
                              self.vipsa.zaber_x.move_absolute(x)
                              self.vipsa.zaber_y.move_absolute(y)

                          # Ensure save directory for this device
                          device_save_dir = os.path.join(save_dir, sample_id, f"device_{int(dev_id)}")
                          os.makedirs(device_save_dir, exist_ok=True)

                          # Execute protocol steps for this device
                          results = self.vipsa.run_protocol(self.protocol_builder.protocol_list_configs,
                                                             sample_no=sample_id,
                                                             device_no=int(dev_id),
                                                             save_directory=device_save_dir,
                                                             SMU=None, stage=None,
                                                             Zaber_x=None, Zaber_y=None,
                                                             top_light=None,
                                                             stop_on_error=False)
                          print(f"Protocol results for device {int(dev_id)}: {results}")

                      except Exception as e:
                          print(f"Error running protocol on device {int(dev_id)}: {e}")

                  print("Grid Protocol Measurement loop finished.")


             print(f"Grid Measurement ({measurement_type}) finished.")

         except ValueError:
              print("Error: Invalid numeric value for grid measurement parameters.")
         except Exception as e:
              print(f"Error during grid measurement: {e}")

    # --- Protocol Handling ---
    def _add_protocol_step(self, values):
        """Open enhanced protocol step editor dialog."""
        result = self.protocol_builder.show_step_editor()
        if result:
            print(result)
            self._update_protocol_display()

    def _update_protocol_display(self):
        """Update the displayed protocol list in the GUI."""
        display_list = self.protocol_builder.get_protocol_display_list()
        self.window['-TEST_LIST-'].update(values=display_list)

    def _edit_protocol_step(self, selected_index):
        """Edit an existing protocol step."""
        if 0 <= selected_index < len(self.protocol_builder.protocol_list_configs):
            step = self.protocol_builder.protocol_list_configs[selected_index]
            editor = ProtocolStepEditor(editing_index=selected_index)
            updated_step = editor.run(step)
            if updated_step:
                self.protocol_builder.protocol_list_configs[selected_index] = updated_step
                print(f"Updated step {selected_index + 1}: {updated_step['type']}")
                self._update_protocol_display()

    def _remove_protocol_step(self):
        """Remove selected protocol step. Double-click or right-click to edit."""
        selected_indices = self.window['-TEST_LIST-'].get_indexes()
        if not selected_indices:
            print("No test selected to remove.")
            return
        selected_index = selected_indices[0]
        result = self.protocol_builder.remove_step(selected_index)
        if result:
            print(result)
            self._update_protocol_display()

    def _clear_protocol(self):
        """Clear all protocol steps."""
        if sg.popup_yes_no('Clear all protocol steps? This cannot be undone.') == 'Yes':
            self.protocol_builder.clear_protocol()
            self._update_protocol_display()
            print("Protocol cleared.")

    # --- Protocol persistence and run handlers ---
    def _save_protocol(self):
        """Save protocol to JSON file."""
        filepath = sg.popup_get_file('Save protocol as', save_as=True, 
                                     file_types=(('JSON Files', '*.json'),), 
                                     default_extension='json')
        if not filepath:
            print('Save cancelled.')
            return
        ok = self.protocol_builder.export_protocol(filepath)
        if ok:
            print(f'Protocol saved to {filepath}')

    def _load_protocol(self):
        """Load protocol from JSON file."""
        filepath = sg.popup_get_file('Load protocol', file_types=(('JSON Files', '*.json'),))
        if not filepath:
            print('Load cancelled.')
            return
        ok = self.protocol_builder.import_protocol(filepath)
        if ok:
            self._update_protocol_display()
            print(f'Protocol loaded with {len(self.protocol_builder.protocol_list_configs)} steps.')

    def _run_protocol_single(self, values):
        """Run protocol on current single device target."""
        if not self.is_equipment_connected:
            print('Error: Equipment not connected.')
            return
        if not self.protocol_builder.protocol_list_configs:
            print('Error: Protocol empty. Add steps using the protocol builder.')
            sg.popup_error('Protocol is empty', 'Please add at least one step to your protocol.')
            return
        
        sample_id = values.get('-MEAS_SAMPLE_ID-', 'sample')
        device_id = values.get('-MEAS_DEVICE_ID-', '0')
        save_dir = values.get('-MEAS_SAVE_FOLDER-', DEFAULT_SAVE_DIRECTORY)
        try:
            device_no = int(device_id)
        except Exception:
            device_no = device_id

        device_save_dir = os.path.join(save_dir, sample_id, f"device_{device_no}")
        os.makedirs(device_save_dir, exist_ok=True)

        print(f'Running protocol on Sample {sample_id}, Device {device_no}...')
        results = self.vipsa.run_protocol(self.protocol_builder.protocol_list_configs, 
                                         sample_no=sample_id, device_no=device_no, 
                                         save_directory=device_save_dir)
        print('Protocol run complete. Results:')
        print(results)


    # =================== Main Loop ===================
    def run(self):
        print("Starting ViPSA GUI...")
        while True:
            event, values = self.window.read(timeout=100) # Add timeout for camera update check

            if event == sg.WIN_CLOSED:
                self._stop_camera_thread() # Stop camera before exit
                self._disconnect_all() # Disconnect equipment
                break

            # --- Handle Camera Feed Update Event ---
            if event == '-UPDATE_CAMERA_FEED-':
                 imgbytes = values[event]
                 self.window['-CAMERA_FEED-'].update(data=imgbytes)
                 self.window['-GRID_CAMERA_FEED-'].update(data=imgbytes) # Update both feeds


            # --- Camera Controls ---
            elif event in ('-START_CAMERA-', '-GRID_START_CAMERA-'):
                self._start_camera_thread()
            elif event in ('-STOP_CAMERA-', '-GRID_STOP_CAMERA-'):
                self._stop_camera_thread()

            # --- Connections ---
            elif event == '-CONNECT_ALL-':      self._connect_all()
            elif event == '-DISCONNECT_ALL-':   self._disconnect_all()
            # Add handlers for individual connections if needed, but Connect All is primary

            # --- Movement ---
            elif event == '-MOVE_X_POS-':       self._handle_arduino_move('X', '-X_STEPS-', 1)
            elif event == '-MOVE_X_NEG-':       self._handle_arduino_move('X', '-X_STEPS-', -1) # Backend handles sign
            elif event == '-MOVE_Y_POS-':       self._handle_arduino_move('Y', '-Y_STEPS-', 1)
            elif event == '-MOVE_Y_NEG-':       self._handle_arduino_move('Y', '-Y_STEPS-', -1)
            elif event == '-MOVE_Z_POS-':       self._handle_arduino_move('Z', '-Z_STEPS-', 1)
            elif event == '-MOVE_Z_NEG-':       self._handle_arduino_move('Z', '-Z_STEPS-', -1)
            elif event == '-SET_ZERO-':
                 if self.vipsa.stage: self.vipsa.stage.set_zero()
                 else: print("Arduino stage not connected.")
            elif event == '-GO_ZERO-':
                 if self.vipsa.stage: self.vipsa.stage.go_to_zero()
                 else: print("Arduino stage not connected.")
            elif event == '-MOVE_ZABER_X_POS-': self._handle_zaber_move('X', '-ZABER_X_DIST-', 1)
            elif event == '-MOVE_ZABER_X_NEG-': self._handle_zaber_move('X', '-ZABER_X_DIST-', -1)
            elif event == '-MOVE_ZABER_Y_POS-': self._handle_zaber_move('Y', '-ZABER_Y_DIST-', 1)
            elif event == '-MOVE_ZABER_Y_NEG-': self._handle_zaber_move('Y', '-ZABER_Y_DIST-', -1)

            # --- Lights ---
            elif event == '-TURN_ON_LIGHTS-':   self._control_lights('on')
            elif event == '-TURN_OFF_LIGHTS-':  self._control_lights('off')

            # --- Grid Creation ---
            elif event == '-CREATE_GRID-':      self._create_grid_csv(values)
            
            # --- Probing Events ---
            
            elif event == '-QUICK_APPROACH-' : self._quick_approach()
            elif event == '-QUICK_ALIGN-' : self._quick_align()

            # --- Measurements ---
            elif event == '-RUN_SINGLE_DCIV_MEAS-': self._run_single_dciv_meas(values)
            elif event == '-RUN_SINGLE_PULSE_MEAS-': self._run_single_pulse_meas(values)
            elif event == '-RUN_GRID_DCIV-':        self._run_grid_measurement(values, 'DCIV')
            elif event == '-RUN_GRID_PULSE-':       self._run_grid_measurement(values, 'PULSE')
            elif event == '-RUN_GRID_PROTOCOL-':    self._run_grid_measurement(values, 'PROTOCOL')
            elif event == '-RUN_MEAS_RESISTANCE-':  self._run_measure_resistance(values)

            # --- Protocol Builder ---
            elif event == '-ADD_TEST-':          self._add_protocol_step(values)
            elif event == '-EDIT_TEST-':         
                selected_indices = self.window['-TEST_LIST-'].get_indexes()
                if selected_indices:
                    self._edit_protocol_step(selected_indices[0])
            elif event == '-REMOVE_TEST-':       self._remove_protocol_step()
            elif event == '-CLEAR_PROTOCOL-':    self._clear_protocol()
            elif event == '-SAVE_PROTOCOL-':     self._save_protocol()
            elif event == '-LOAD_PROTOCOL-':     self._load_protocol()
            elif event == '-RUN_PROTOCOL_SINGLE-': self._run_protocol_single(values)


        self.window.close()
        print("ViPSA GUI closed.")


if __name__ == '__main__':
    gui = VipsaGUI()
    gui.run()