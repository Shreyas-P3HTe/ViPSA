# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:19:54 2025

@author: shrey

Standalone Measurement GUI
Combines Listmaker, Protocol Builder, and Manual Measurement execution
for use with external SMU and manual probing.
"""

import os
import PySimpleGUI as sg
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import threading # To prevent GUI freeze during measurement

# Import necessary backend classes
from Source_Measure_Unit import KeysightSMU # Make sure this file is accessible
from Datahandling import Data_Handler # Make sure this file is accessible
# Listmaker functions will be integrated directly or called if Listmaker2.py is importable

# --- Default Paths and Settings ---
DEFAULT_SAVE_DIRECTORY = "./manual_probe_data" # Adjust as needed
# Leave sweep/pulse paths empty initially, user must generate or select
DEFAULT_SWEEP_PATH = ""
DEFAULT_PULSE_PATH = ""

# --- Listmaker Functions (from Listmaker2.py - integrated for standalone) ---
class ListmakerFunctions:
    @staticmethod
    def generate_voltage_data(forward_voltage, reset_voltage, step_voltage, timer_delay, forming_cycle, forming_voltage, cycles):
        voltages = []
        times = []
        current_time = 0.0 # Ensure float

        # Forming cycle if needed
        if forming_cycle and forming_voltage is not None:
            voltage = 0.0
            while voltage <= forming_voltage:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage += step_voltage

            voltage -= step_voltage # Go back to peak
            while voltage >= 0:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage -= step_voltage

            # Immediate reset sweep after forming cycle
            voltage = 0.0
            while voltage >= reset_voltage:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage -= step_voltage

            voltage += step_voltage # Go back to peak
            while voltage <= 0:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage += step_voltage

        # Regular cycles
        num_regular_cycles = cycles if not forming_cycle else cycles -1 # Adjust if forming was done
        if num_regular_cycles < 0: num_regular_cycles = 0

        for _ in range(num_regular_cycles):
            # Forward voltage cycle
            voltage = 0.0
            while voltage <= forward_voltage:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage += step_voltage

            voltage -= step_voltage # Go back to peak
            while voltage >= 0:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage -= step_voltage

            # Reset voltage cycle
            voltage = 0.0
            while voltage >= reset_voltage:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage -= step_voltage

            voltage += step_voltage # Go back to peak
            while voltage <= 0:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage += step_voltage

        # Ensure the list ends at 0V if needed
        if voltages and voltages[-1] != 0:
             voltages.append(0.0)
             times.append(current_time)


        return times, voltages

    @staticmethod
    def generate_pulsing_data(write_pulses, write_voltage, write_width, read_pulses, read_voltage, read_width, erase_pulses, erase_voltage, erase_width, cycles):
        voltages = []
        times = []
        current_time = 0.0 # Ensure float

        for _ in range(int(cycles)):
            # Write pulses
            for _ in range(int(write_pulses)):
                voltages.append(write_voltage)
                times.append(current_time)
                current_time += write_width
                voltages.append(0) # Return to 0V between pulses
                times.append(current_time)
                # current_time += write_width # Optional hold time at 0V

            # Read pulses
            for _ in range(int(read_pulses)):
                voltages.append(read_voltage)
                times.append(current_time)
                current_time += read_width
                voltages.append(0)
                times.append(current_time)
                # current_time += read_width

            # Erase pulses
            for _ in range(int(erase_pulses)):
                voltages.append(erase_voltage)
                times.append(current_time)
                current_time += erase_width
                voltages.append(0)
                times.append(current_time)
                # current_time += erase_width

            # Read pulses again
            for _ in range(int(read_pulses)):
                voltages.append(read_voltage)
                times.append(current_time)
                current_time += read_width
                voltages.append(0)
                times.append(current_time)
                # current_time += read_width

        return times, voltages

    @staticmethod
    def save_list_to_csv(times, voltages, filename):
        """Saves the generated time and voltage list to a CSV file."""
        if not filename:
            print("Error: No filename provided for saving.")
            return False
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Time (s)', 'Voltage (V)']) # Header consistent with SMU code expectations
                for t, v in zip(times, voltages):
                    csvwriter.writerow([f"{t:.6f}", f"{v:.6f}"]) # Format for consistency
            print(f"List saved successfully to {filename}")
            return True
        except Exception as e:
            print(f"Error saving list to CSV {filename}: {e}")
            return False

    @staticmethod
    def plot_list_data(times, voltages, title="Generated List"):
        """Plots the generated time vs voltage list."""
        plt.figure(figsize=(10, 5))
        plt.plot(times, voltages, marker='.', linestyle='-', markersize=4)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show() # Use standard blocking show for list visualization


# --- Plotting Modifications ---
class PlottingHandler(Data_Handler): # Inherit from Data_Handler

    def show_plot_briefly(self, csvpath, sample_id="NA", device_id="NA", timeout=5):
        """Shows DCIV plot briefly and closes."""
        print(f"Plotting DCIV for {sample_id}-{device_id} (will close after {timeout}s)...")
        try:
            fig, ax = plt.subplots(figsize=(8, 6)) # Create figure and axes explicitly

            df = pd.read_csv(csvpath)
            df['Current (A)'] = df['Current (A)'].abs() # Use absolute current

            colormap = cm.get_cmap('viridis') # Use a different colormap maybe
            num_cycles = df['Voltage (V)'].eq(0).cumsum().max() # Estimate number of cycles
            colors = [colormap(i / max(1, num_cycles)) for i in range(max(1, num_cycles))]

            start_index = 0
            color_idx = 0
            legend_labels = []

            for i in range(1, len(df)):
                 # Cycle start condition (customize if needed)
                if df['Voltage (V)'].iloc[i-1] == 0 and (df['Voltage (V)'].iloc[i] != 0 or i == 1):
                     if start_index != i-1 or i == 1: # Plot segment
                         if i > 1: # Avoid plotting single point at start
                             ax.plot(df['Voltage (V)'].iloc[start_index:i],
                                     df['Current (A)'].iloc[start_index:i],
                                     linestyle='-', marker='.', markersize=3,
                                     color=colors[min(color_idx, len(colors)-1)],
                                     label=f'Cycle {color_idx + 1}')
                             legend_labels.append(f'Cycle {color_idx + 1}')
                             color_idx += 1
                     start_index = i

            # Plot the last cycle
            if start_index < len(df):
                ax.plot(df['Voltage (V)'].iloc[start_index:],
                        df['Current (A)'].iloc[start_index:],
                        linestyle='-', marker='.', markersize=3,
                        color=colors[min(color_idx, len(colors)-1)],
                        label=f'Cycle {color_idx + 1}')
                legend_labels.append(f'Cycle {color_idx + 1}')

            ax.set_ylabel('Current (|A|, log scale)')
            ax.set_xlabel('Voltage (V)')
            ax.set_yscale('log')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend(title="Cycles", fontsize='small')
            ax.set_title(f"DCIV: {sample_id} - Dev {device_id}")

            plt.ion() # Turn interactive mode on
            plt.show(block=False) # Show plot without blocking
            plt.pause(timeout) # Keep plot window open for 'timeout' seconds
            plt.close(fig) # Close the specific figure
            plt.ioff() # Turn interactive mode off
            print("Plot closed.")

        except Exception as e:
            print(f"Error plotting DCIV {csvpath}: {e}")
            # Ensure plot is closed if error occurs during display
            plt.close('all')
            plt.ioff()

    def show_pulse_briefly(self, csvpath, sample_id="NA", device_id="NA", timeout=5):
        """Shows Pulse plot briefly and closes."""
        print(f"Plotting Pulse for {sample_id}-{device_id} (will close after {timeout}s)...")
        try:
            fig, ax = plt.subplots(figsize=(12, 6)) # Create figure and axes explicitly
            df = pd.read_csv(csvpath)

            # Basic pulse plot - plot current vs index or time
            ax.plot(df.index, df['Current (A)'], marker='.', linestyle='-', markersize=3, label='Current')
            # Optional: Add voltage overlay on secondary axis
            ax2 = ax.twinx()
            ax2.plot(df.index, df['Voltage (V)'], marker='.', linestyle=':', color='red', markersize=2, label='Voltage (V)')
            ax2.set_ylabel('Voltage (V)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            ax.set_xlabel("Measurement Index")
            ax.set_ylabel("Current (A), log scale")
            ax.set_yscale("log") # Use log scale for current
            ax.set_title(f"Pulsed Measurement: {sample_id} - Dev {device_id}")
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            # Combine legends if using twin axes
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='best', fontsize='small')

            plt.ion()
            plt.show(block=False)
            plt.pause(timeout)
            plt.close(fig)
            plt.ioff()
            print("Plot closed.")

        except Exception as e:
            print(f"Error plotting Pulse {csvpath}: {e}")
            plt.close('all')
            plt.ioff()

    def save_manual_file(self, data, data_name, sample_id, device_id, save_directory=None):
        """Saves data from manual measurements (no Z_pos/Contact Current)."""
        if save_directory is None:
            save_directory = DEFAULT_SAVE_DIRECTORY

        # Create a unique folder for this sample/device
        directory_path = os.path.join(save_directory, str(sample_id), str(device_id), str(data_name))
        os.makedirs(directory_path, exist_ok=True)

        # Create a filename (e.g., based on timestamp)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(directory_path, f"{data_name}_{timestamp}.csv")

        try:
            # Assuming data is list of tuples/lists: (time, voltage, current)
            df = pd.DataFrame(data, columns=['Time(T)', 'Voltage (V)', 'Current (A)'])
            df.to_csv(file_path, index=True, index_label="Index") # Save with index
            print(f"Data saved successfully to {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving data to {file_path}: {e}")
            return None

# --- Main GUI Class ---
class StandaloneTesterGUI:
    def __init__(self):
        sg.theme('Reddit')
        self.smu = None # Initialize SMU object later upon connection
        self.plot_handler = PlottingHandler() # Use the modified handler
        self.list_maker = ListmakerFunctions() # Use the static methods
        self.is_smu_connected = False
        self.protocol_list_configs = []
        self.window = self.create_window()

        # Temporary storage for generated lists before saving
        self.generated_sweep_list = None
        self.generated_pulse_list = None


    def create_window(self):

        # --- List Generation Tab ---
        list_gen_iv_layout = [
            [sg.Text('--- I-V Sweep List Generation ---', font='_ 11')],
            [sg.Text('Forward V:', size=(12,1)), sg.InputText('1.5', key='-IV_FWD_V-', size=(8,1)),
             sg.Text('Reset V (-ve):', size=(12,1)), sg.InputText('-1.5', key='-IV_RST_V-', size=(8,1))],
            [sg.Text('Step V:', size=(12,1)), sg.InputText('0.05', key='-IV_STEP_V-', size=(8,1)),
             sg.Text('Step Delay (s):', size=(12,1)), sg.InputText('0.001', key='-IV_DELAY-', size=(8,1))],
            [sg.Text('Cycles:', size=(12,1)), sg.InputText('2', key='-IV_CYCLES-', size=(8,1))],
            [sg.Checkbox('Forming Cycle?', key='-IV_FORM_CHK-', enable_events=True),
             sg.Text('Forming V:', size=(10,1)), sg.InputText('', key='-IV_FORM_V-', size=(8,1), disabled=True)],
             [sg.Button('Visualize IV List', key='-VIZ_IV-'), sg.Button('Save IV List As...', key='-SAVE_IV-')]
        ]
        list_gen_pulse_layout = [
            [sg.Text('--- Pulsing List Generation ---', font='_ 11')],
            [sg.Text('Set (#V, Width):', size=(15,1)),
             sg.InputText('1', key='-PULSE_SET_N-', size=(5,1)), sg.InputText('1.5', key='-PULSE_SET_V-', size=(8,1)), sg.InputText('0.001', key='-PULSE_SET_W-', size=(8,1))],
             [sg.Text('Read (#V, Width):', size=(15,1)),
             sg.InputText('1', key='-PULSE_READ_N-', size=(5,1)), sg.InputText('0.1', key='-PULSE_READ_V-', size=(8,1)), sg.InputText('0.001', key='-PULSE_READ_W-', size=(8,1))],
            [sg.Text('Reset (#, V, Width):', size=(15,1)),
             sg.InputText('1', key='-PULSE_ERASE_N-', size=(5,1)), sg.InputText('-1.5', key='-PULSE_ERASE_V-', size=(8,1)), sg.InputText('0.001', key='-PULSE_ERASE_W-', size=(8,1))],
            [sg.Text('Cycles:', size=(15,1)), sg.InputText('10', key='-PULSE_CYCLES-', size=(5,1))],
             [sg.Button('Visualize Pulse List', key='-VIZ_PULSE-'), sg.Button('Save Pulse List As...', key='-SAVE_PULSE-')]
        ]

        list_generation_tab = [
             [sg.Column(list_gen_iv_layout)],
             [sg.HSeparator()],
             [sg.Column(list_gen_pulse_layout)]
        ]

        # --- Manual Measurement Tab ---
        manual_meas_layout = [
            [sg.Text("--- SMU Connection ---", font='_ 11')],
             [sg.Button("Connect SMU", key='-CONNECT_SMU-'), sg.Button("Disconnect SMU", key='-DISCONNECT_SMU-'),
              sg.Text("Status: Disconnected", key='-SMU_STATUS-', text_color='red')],
             [sg.HSeparator()],
             [sg.Text("--- Measurement Info ---", font='_ 11')],
             [sg.Text("Sample ID:", size=(10,1)), sg.InputText("SampleX", key='-MAN_SAMPLE_ID-', size=(15, 1)),
              sg.Text("Device #:", size=(8,1)), sg.InputText("Dev1", key='-MAN_DEVICE_ID-', size=(10, 1))],
             [sg.Text("Save Folder:", size=(10,1)), sg.InputText(DEFAULT_SAVE_DIRECTORY, key='-MAN_SAVE_FOLDER-', size=(40, 1)), sg.FolderBrowse()],
             [sg.HSeparator()],
             [sg.Text("--- Run DCIV from File ---", font='_ 11')],
             [sg.Text("Sweep File:", size=(15,1)), sg.InputText(DEFAULT_SWEEP_PATH, key='-MAN_SWEEP_PATH-', size=(40, 1)), sg.FileBrowse()],
             [sg.Text("Pos Comp (A):", size=(15,1)), sg.InputText('0.001', key='-MAN_POS_COMP-', size=(10, 1)),
              sg.Text("Neg Comp (A):", size=(15,1)), sg.InputText('0.01', key='-MAN_NEG_COMP-', size=(10, 1))],
             [sg.Text("Sweep Delay (s, opt):", size=(15,1)), sg.InputText('0.0001', key='-MAN_SWEEP_DELAY-', size=(10,1))],
             [sg.Text("Plot Timeout (s):", size=(15,1)), sg.InputText('5', key='-MAN_DCIV_PLOT_TIMEOUT-', size=(5,1)), sg.Checkbox('Plot?', key='-MAN_DCIV_PLOT_CHK-', default=True)],
             [sg.Button("Run DCIV Measurement", key='-RUN_MAN_DCIV-')],
             [sg.HSeparator()],
              [sg.Text("--- Run Pulse from File ---", font='_ 11')],
             [sg.Text("Pulse File:", size=(15,1)), sg.InputText(DEFAULT_PULSE_PATH, key='-MAN_PULSE_PATH-', size=(40, 1)), sg.FileBrowse()],
             [sg.Text("Compliance (A):", size=(15,1)), sg.InputText('0.01', key='-MAN_PULSE_COMP-', size=(10, 1))],
             [sg.Text("Pulse Width (s, opt):", size=(15,1)), sg.InputText('', key='-MAN_PULSE_WIDTH-', size=(10,1))],
             [sg.Text("Plot Timeout (s):", size=(15,1)), sg.InputText('5', key='-MAN_PULSE_PLOT_TIMEOUT-', size=(5,1)), sg.Checkbox('Plot?', key='-MAN_PULSE_PLOT_CHK-', default=True)],
             [sg.Button("Run Pulsed Measurement", key='-RUN_MAN_PULSE-')]
        ]

        # --- Protocol Builder Tab ---
        protocol_editor_layout = [
             [sg.Text("Available Test Types:")],
             [sg.Combo(['DCIV from File', 'Pulse from File'], key='-PROTO_TEST_TYPE-', readonly=True)],
             [sg.Button('Add Test to Protocol', key='-PROTO_ADD-')],
             [sg.Button('Remove Selected Test', key='-PROTO_REMOVE-')],
             [sg.Button('Clear Protocol', key='-PROTO_CLEAR-')],
        ]
        protocol_viewer_layout = [
             [sg.Text("Protocol Steps:")],
             [sg.Listbox(values=[], size=(70, 10), key='-PROTO_LISTBOX-')],
             [sg.Text("--- Protocol Execution ---", font='_ 11')],
             [sg.Text("Sample ID:", size=(10,1)), sg.InputText("ProtoSample", key='-PROTO_SAMPLE_ID-', size=(15, 1)),
              sg.Text("Device #:", size=(8,1)), sg.InputText("ProtoDev", key='-PROTO_DEVICE_ID-', size=(10, 1))],
             [sg.Text("Save Folder:", size=(10,1)), sg.InputText(DEFAULT_SAVE_DIRECTORY, key='-PROTO_SAVE_FOLDER-', size=(40, 1)), sg.FolderBrowse()],
             [sg.Text("Plot Timeout (s):", size=(15,1)), sg.InputText('3', key='-PROTO_PLOT_TIMEOUT-', size=(5,1)), sg.Checkbox('Plot Steps?', key='-PROTO_PLOT_CHK-', default=False)],
             [sg.Button('Run Protocol', key='-PROTO_RUN-')]
        ]
        protocol_tab = [
             [sg.Column(protocol_editor_layout), sg.VSeperator(), sg.Column(protocol_viewer_layout)]
        ]

        # --- Output Log Tab ---
        output_tab = [
             [sg.Text("--- Log Output ---", font='_ 11')],
             [sg.Multiline(size=(100, 20), key="-OUTPUT-", autoscroll=True, disabled=True, reroute_stdout=True, reroute_stderr=True)]
        ]

        # --- Main Layout ---
        layout = [[sg.TabGroup([
                     [sg.Tab('List Generation', list_generation_tab),
                      sg.Tab('Manual Measurement', manual_meas_layout),
                      sg.Tab('Protocol Builder', protocol_tab),
                      sg.Tab('Output Log', output_tab)]
                 ])]]

        return sg.Window("Standalone Measurement Controller", layout, finalize=True)

    # --- Helper Functions ---
    def _get_float(self, key, default=None):
        """Safely get a float value from GUI input."""
        try:
            val_str = self.window[key].get()
            return float(val_str) if val_str else default
        except ValueError:
            print(f"Warning: Invalid float input for key '{key}'. Using default: {default}")
            return default

    def _get_int(self, key, default=None):
        """Safely get an integer value from GUI input."""
        try:
            val_str = self.window[key].get()
            return int(val_str) if val_str else default
        except ValueError:
             print(f"Warning: Invalid integer input for key '{key}'. Using default: {default}")
             return default

    def _validate_file(self, key, file_type="File"):
         """Checks if the file path in the input element exists."""
         fpath = self.window[key].get()
         if not fpath or not os.path.exists(fpath):
              print(f"Error: {file_type} path is invalid or file does not exist: '{fpath}'")
              sg.popup_error(f"{file_type} path is invalid or file does not exist:\n{fpath}", title=f"Invalid {file_type} Path")
              return None
         return fpath

    def _run_measurement_in_thread(self, target_func, args_tuple):
        """Runs a measurement function in a thread to avoid GUI freeze."""
        thread = threading.Thread(target=target_func, args=args_tuple, daemon=True)
        thread.start()
        # We don't join here, the function handles its own completion message/plotting

    # --- Event Handlers ---
    def _handle_connect_smu(self):
        if self.is_smu_connected:
             print("SMU already connected.")
             return
        print("Connecting to SMU...")
        try:
            self.smu = KeysightSMU(0) # Assuming device 0
            self.is_smu_connected = True
            self.window['-SMU_STATUS-'].update('Status: Connected', text_color='green')
            print(f"SMU connected: {self.smu.address}") # Print address
        except Exception as e:
            self.is_smu_connected = False
            self.smu = None
            self.window['-SMU_STATUS-'].update('Status: Connection Failed', text_color='red')
            print(f"SMU Connection Error: {e}")
            sg.popup_error(f"SMU Connection Error:\n{e}", title="Connection Error")

    def _handle_disconnect_smu(self):
        if not self.is_smu_connected:
            print("SMU already disconnected.")
            return
        print("Disconnecting SMU...")
        # For pyvisa, closing isn't strictly necessary here as KeysightSMU might handle it
        # but we reset the state
        self.smu = None
        self.is_smu_connected = False
        self.window['-SMU_STATUS-'].update('Status: Disconnected', text_color='red')
        print("SMU disconnected.")

    def _handle_list_gen_form_check(self, values):
         """Enable/disable forming voltage input based on checkbox."""
         is_checked = values['-IV_FORM_CHK-']
         self.window['-IV_FORM_V-'].update(disabled=not is_checked)
         if not is_checked:
             self.window['-IV_FORM_V-'].update('') # Clear value if disabled

    def _handle_visualize_iv(self, values):
         print("Generating IV list for visualization...")
         try:
             fwd_v = self._get_float('-IV_FWD_V-', 1.5)
             rst_v = self._get_float('-IV_RST_V-', -1.5)
             step_v = self._get_float('-IV_STEP_V-', 0.05)
             delay = self._get_float('-IV_DELAY-', 0.001)
             cycles = self._get_int('-IV_CYCLES-', 2)
             use_form = values['-IV_FORM_CHK-']
             form_v = self._get_float('-IV_FORM_V-', None) if use_form else None

             if step_v <= 0 or delay <= 0:
                 print("Error: Step Voltage and Step Delay must be positive.")
                 sg.popup_error("Step Voltage and Step Delay must be positive.")
                 return

             times, voltages = self.list_maker.generate_voltage_data(
                 fwd_v, rst_v, step_v, delay, use_form, form_v, cycles
             )
             self.generated_sweep_list = (times, voltages) # Store for saving
             print(f"IV list generated ({len(times)} points). Plotting...")
             self.list_maker.plot_list_data(times, voltages, "Generated IV Sweep List")
         except Exception as e:
             print(f"Error generating/visualizing IV list: {e}")
             sg.popup_error(f"Error generating/visualizing IV list:\n{e}")

    def _handle_save_iv(self):
        if not self.generated_sweep_list:
            print("No IV list generated yet. Please Visualize first.")
            sg.popup_error("No IV list generated yet. Please Visualize first.")
            return

        save_path = sg.popup_get_file('Save IV Sweep List As...', save_as=True,
                                      no_window=True, file_types=(("CSV Files", "*.csv"),),
                                      default_path="sweep_list.csv")
        if save_path:
            times, voltages = self.generated_sweep_list
            if self.list_maker.save_list_to_csv(times, voltages, save_path):
                self.window['-MAN_SWEEP_PATH-'].update(save_path) # Update manual path


    def _handle_visualize_pulse(self, values):
        print("Generating Pulse list for visualization...")
        try:
            set_n = self._get_int('-PULSE_SET_N-', 1)
            set_v = self._get_float('-PULSE_SET_V-', 1.5)
            set_w = self._get_float('-PULSE_SET_W-', 0.001)
            read_n = self._get_int('-PULSE_READ_N-', 1)
            read_v = self._get_float('-PULSE_READ_V-', 0.1)
            read_w = self._get_float('-PULSE_READ_W-', 0.001)
            erase_n = self._get_int('-PULSE_ERASE_N-', 1)
            erase_v = self._get_float('-PULSE_ERASE_V-', -1.5)
            erase_w = self._get_float('-PULSE_ERASE_W-', 0.001)
            cycles = self._get_int('-PULSE_CYCLES-', 10)

            if any(w <= 0 for w in [set_w, read_w, erase_w]):
                 print("Error: Pulse widths must be positive.")
                 sg.popup_error("Pulse widths must be positive.")
                 return

            times, voltages = self.list_maker.generate_pulsing_data(
                 set_n, set_v, set_w, read_n, read_v, read_w, erase_n, erase_v, erase_w, cycles
            )
            self.generated_pulse_list = (times, voltages) # Store for saving
            print(f"Pulse list generated ({len(times)} points). Plotting...")
            self.list_maker.plot_list_data(times, voltages, "Generated Pulse List")
        except Exception as e:
            print(f"Error generating/visualizing Pulse list: {e}")
            sg.popup_error(f"Error generating/visualizing Pulse list:\n{e}")

    def _handle_save_pulse(self):
        if not self.generated_pulse_list:
            print("No Pulse list generated yet. Please Visualize first.")
            sg.popup_error("No Pulse list generated yet. Please Visualize first.")
            return

        save_path = sg.popup_get_file('Save Pulse List As...', save_as=True,
                                      no_window=True, file_types=(("CSV Files", "*.csv"),),
                                      default_path="pulse_list.csv")
        if save_path:
            times, voltages = self.generated_pulse_list
            if self.list_maker.save_list_to_csv(times, voltages, save_path):
                 self.window['-MAN_PULSE_PATH-'].update(save_path) # Update manual path

    # --- Manual Measurement Execution ---
    def _execute_manual_dciv(self, values):
        if not self.is_smu_connected:
            print("Error: SMU not connected.")
            sg.popup_error("SMU is not connected. Please connect first.", title="Connection Error")
            return

        sweep_path = self._validate_file('-MAN_SWEEP_PATH-', "Sweep")
        if not sweep_path: return

        try:
            pos_comp = self._get_float('-MAN_POS_COMP-', 0.001)
            neg_comp = self._get_float('-MAN_NEG_COMP-', 0.01)
            sweep_delay = self._get_float('-MAN_SWEEP_DELAY-', None)
            sample_id = values['-MAN_SAMPLE_ID-']
            device_id = values['-MAN_DEVICE_ID-']
            save_dir = values['-MAN_SAVE_FOLDER-']
            do_plot = values['-MAN_DCIV_PLOT_CHK-']
            plot_timeout = self._get_int('-MAN_DCIV_PLOT_TIMEOUT-', 5)

            if not sample_id or not device_id:
                 print("Error: Please provide Sample ID and Device ID.")
                 sg.popup_error("Please provide Sample ID and Device ID.")
                 return

            print(f"\nStarting Manual DCIV for {sample_id}-{device_id} using {os.path.basename(sweep_path)}...")

            # Define the target function for the thread
            def measurement_task():
                print("Measurement thread started...")
                sweep_data, res_data = self.smu.list_IV_sweep_split(
                    csv_path=sweep_path,
                    pos_compliance=pos_comp,
                    neg_compliance=neg_comp,
                    delay=sweep_delay,
                    adr=self.smu.address
                )
                print("SMU sweep finished.")

                saved_sweep_path = None
                saved_res_path = None
                if sweep_data is not None and len(sweep_data) > 0:
                    print("Sweep successful, saving data...")
                    saved_sweep_path = self.plot_handler.save_manual_file(sweep_data, "Sweep", sample_id, device_id, save_dir)
                    if saved_sweep_path and do_plot:
                         # Run plotting in the main thread after task completion if possible,
                         # otherwise, try brief plot here (might have issues depending on backend)
                         try:
                              self.plot_handler.show_plot_briefly(saved_sweep_path, sample_id, device_id, plot_timeout)
                         except Exception as plot_err:
                              print(f"Error during brief plot: {plot_err}")
                else:
                    print("Sweep measurement failed or returned no data.")

                if res_data is not None and len(res_data) > 0:
                     print("Resistance data obtained, saving...")
                     saved_res_path = self.plot_handler.save_manual_file(res_data, "Resistance", sample_id, device_id, save_dir)
                     # Optionally plot resistance data briefly too
                     # if saved_res_path and do_plot: self.plot_handler.show_resistance_briefly(...)
                else:
                     print("Resistance measurement failed or returned no data.")
                print(f"Manual DCIV task finished for {sample_id}-{device_id}.")

            # Run the task in a thread
            self._run_measurement_in_thread(measurement_task, ())

        except Exception as e:
             print(f"Error setting up manual DCIV: {e}")
             sg.popup_error(f"Error setting up manual DCIV:\n{e}")


    def _execute_manual_pulse(self, values):
        if not self.is_smu_connected:
            print("Error: SMU not connected.")
            sg.popup_error("SMU is not connected. Please connect first.", title="Connection Error")
            return

        pulse_path = self._validate_file('-MAN_PULSE_PATH-', "Pulse")
        if not pulse_path: return

        try:
            compliance = self._get_float('-MAN_PULSE_COMP-', 0.01)
            pulse_width = self._get_float('-MAN_PULSE_WIDTH-', None)
            sample_id = values['-MAN_SAMPLE_ID-']
            device_id = values['-MAN_DEVICE_ID-']
            save_dir = values['-MAN_SAVE_FOLDER-']
            do_plot = values['-MAN_PULSE_PLOT_CHK-']
            plot_timeout = self._get_int('-MAN_PULSE_PLOT_TIMEOUT-', 5)

            if not sample_id or not device_id:
                 print("Error: Please provide Sample ID and Device ID.")
                 sg.popup_error("Please provide Sample ID and Device ID.")
                 return

            print(f"\nStarting Manual Pulse for {sample_id}-{device_id} using {os.path.basename(pulse_path)}...")

            # Define the target function for the thread
            def measurement_task():
                print("Measurement thread started...")
                pulse_data = self.smu.pulsed_measurement(
                    csv_path=pulse_path,
                    current_compliance=compliance,
                    set_width=pulse_width,
                    adr=self.smu.address
                )
                print("SMU pulse finished.")

                if pulse_data is not None and len(pulse_data) > 0:
                    print("Pulse successful, saving data...")
                    saved_file_path = self.plot_handler.save_manual_file(pulse_data, "Pulse", sample_id, device_id, save_dir)
                    if saved_file_path and do_plot:
                        try:
                             self.plot_handler.show_pulse_briefly(saved_file_path, sample_id, device_id, plot_timeout)
                        except Exception as plot_err:
                             print(f"Error during brief plot: {plot_err}")
                else:
                    print("Pulse measurement failed or returned no data.")
                print(f"Manual Pulse task finished for {sample_id}-{device_id}.")

             # Run the task in a thread
            self._run_measurement_in_thread(measurement_task, ())

        except Exception as e:
            print(f"Error setting up manual pulse: {e}")
            sg.popup_error(f"Error setting up manual pulse:\n{e}")

    # --- Protocol Execution ---
    def _add_protocol_step(self, values):
        test_type = values['-PROTO_TEST_TYPE-']
        if not test_type:
            print("Please select a test type.")
            sg.popup_error("Please select a test type.")
            return

        # Open popup to configure this step
        # For simplicity, popup asks for file path and main params
        params = {}
        layout = []
        if test_type == 'DCIV from File':
             layout = [[sg.Text("Sweep File:"), sg.Input(key='-PATH-'), sg.FileBrowse()],
                       [sg.Text("Pos Comp:"), sg.Input('0.001', key='-POS_COMP-')],
                       [sg.Text("Neg Comp:"), sg.Input('0.01', key='-NEG_COMP-')],
                       [sg.Text("Sweep Delay (opt):"), sg.Input(key='-DELAY-')],
                       [sg.Submit(), sg.Cancel()]]
             popup_title = "Configure DCIV Step"
        elif test_type == 'Pulse from File':
             layout = [[sg.Text("Pulse File:"), sg.Input(key='-PATH-'), sg.FileBrowse()],
                       [sg.Text("Compliance:"), sg.Input('0.01', key='-COMP-')],
                       [sg.Text("Pulse Width (opt):"), sg.Input(key='-WIDTH-')],
                       [sg.Submit(), sg.Cancel()]]
             popup_title = "Configure Pulse Step"

        if not layout:
            print(f"Configuration popup not defined for: {test_type}")
            return

        popup_window = sg.Window(popup_title, layout)
        event, popup_values = popup_window.read()
        popup_window.close()

        if event != 'Submit': return

        # Validate and store config
        config = {'type': test_type}
        path = popup_values['-PATH-']
        if not path or not os.path.exists(path):
             print(f"Error: Invalid file path provided: {path}")
             sg.popup_error(f"Invalid file path provided:\n{path}")
             return

        try:
             if test_type == 'DCIV from File':
                  params['sweep_path'] = path
                  params['pos_compl'] = float(popup_values['-POS_COMP-'])
                  params['neg_compl'] = float(popup_values['-NEG_COMP-'])
                  params['sweep_delay'] = float(popup_values['-DELAY-']) if popup_values['-DELAY-'] else None
             elif test_type == 'Pulse from File':
                  params['pulse_path'] = path
                  params['compliance'] = float(popup_values['-COMP-'])
                  params['pulse_width'] = float(popup_values['-WIDTH-']) if popup_values['-WIDTH-'] else None

             config['params'] = params
             # Add display string to listbox
             display_string = f"{test_type}: {os.path.basename(path)}"
             self.protocol_list_configs.append(config)
             current_display = self.window['-PROTO_LISTBOX-'].get_list_values()
             current_display.append(display_string)
             self.window['-PROTO_LISTBOX-'].update(values=current_display)
             print(f"Added protocol step: {display_string}")

        except ValueError:
             print("Error: Invalid numeric parameter in popup.")
             sg.popup_error("Invalid numeric parameter entered.")
        except Exception as e:
             print(f"Error adding protocol step: {e}")


    def _remove_protocol_step(self):
        selected_indices = self.window['-PROTO_LISTBOX-'].get_indexes()
        if not selected_indices:
            print("No protocol step selected to remove.")
            return
        selected_index = selected_indices[0]
        try:
            current_display = list(self.window['-PROTO_LISTBOX-'].get_list_values())
            removed_display = current_display.pop(selected_index)
            removed_config = self.protocol_list_configs.pop(selected_index)
            self.window['-PROTO_LISTBOX-'].update(values=current_display)
            print(f"Removed step: {removed_display}")
        except IndexError:
            print("Error: Invalid index selected.")

    def _clear_protocol(self):
        self.protocol_list_configs = []
        self.window['-PROTO_LISTBOX-'].update(values=[])
        print("Protocol cleared.")

    def _execute_protocol(self, values):
        if not self.is_smu_connected:
            print("Error: SMU not connected.")
            sg.popup_error("SMU is not connected. Please connect first.", title="Connection Error")
            return

        if not self.protocol_list_configs:
             print("Error: Protocol is empty.")
             sg.popup_error("Protocol is empty. Please add steps.", title="Empty Protocol")
             return

        try:
            sample_id = values['-PROTO_SAMPLE_ID-']
            device_id = values['-PROTO_DEVICE_ID-']
            save_dir = values['-PROTO_SAVE_FOLDER-']
            do_plot = values['-PROTO_PLOT_CHK-']
            plot_timeout = self._get_int('-PROTO_PLOT_TIMEOUT-', 3)

            if not sample_id or not device_id:
                 print("Error: Please provide Sample ID and Device ID for protocol.")
                 sg.popup_error("Please provide Sample ID and Device ID for protocol.")
                 return

            protocol_steps = list(self.protocol_list_configs) # Copy list for execution thread
            print(f"\nStarting Protocol for {sample_id}-{device_id} ({len(protocol_steps)} steps)...")

            # --- Define threaded task ---
            def protocol_task():
                 print("Protocol thread started...")
                 step_num = 0
                 for step_config in protocol_steps:
                      step_num += 1
                      step_type = step_config.get('type')
                      params = step_config.get('params', {})
                      print(f"\n--- Executing Step {step_num}: {step_type} ---")
                      print(f"Params: {params}")

                      saved_file_path = None # Track saved file for plotting

                      try:
                          if step_type == 'DCIV from File':
                              sweep_data, res_data = self.smu.list_IV_sweep_split(
                                   csv_path=params['sweep_path'],
                                   pos_compliance=params['pos_compl'],
                                   neg_compliance=params['neg_compl'],
                                   delay=params.get('sweep_delay'), # Use get for optional
                                   adr=self.smu.address
                              )
                              if sweep_data is not None and len(sweep_data) > 0:
                                   saved_file_path = self.plot_handler.save_manual_file(sweep_data, f"Step{step_num}_Sweep", sample_id, device_id, save_dir)
                              else: print("DCIV step failed or no data.")
                          elif step_type == 'Pulse from File':
                               pulse_data = self.smu.pulsed_measurement(
                                    csv_path=params['pulse_path'],
                                    current_compliance=params['compliance'],
                                    set_width=params.get('pulse_width'), # Use get for optional
                                    adr=self.smu.address
                               )
                               if pulse_data is not None and len(pulse_data) > 0:
                                    saved_file_path = self.plot_handler.save_manual_file(pulse_data, f"Step{step_num}_Pulse", sample_id, device_id, save_dir)
                               else: print("Pulse step failed or no data.")

                          # Plot if requested and data was saved
                          if do_plot and saved_file_path:
                                if step_type == 'DCIV from File':
                                     self.plot_handler.show_plot_briefly(saved_file_path, sample_id, f"{device_id}_Step{step_num}", plot_timeout)
                                elif step_type == 'Pulse from File':
                                     self.plot_handler.show_pulse_briefly(saved_file_path, sample_id, f"{device_id}_Step{step_num}", plot_timeout)

                          time.sleep(0.2) # Small delay between steps

                      except Exception as step_err:
                           print(f"Error during protocol step {step_num} ({step_type}): {step_err}")
                           # Optionally decide whether to continue or abort protocol on error
                           # break # Uncomment to stop protocol on error

                 print(f"\nProtocol finished for {sample_id}-{device_id}.")
            # --- End of threaded task ---

            self._run_measurement_in_thread(protocol_task, ())

        except Exception as e:
             print(f"Error setting up protocol execution: {e}")
             sg.popup_error(f"Error setting up protocol execution:\n{e}")

    # --- Main Loop ---
    def run(self):
        print("Starting Standalone Measurement GUI...")
        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                if self.is_smu_connected: self._handle_disconnect_smu()
                break

            # --- Connections ---
            elif event == '-CONNECT_SMU-':      self._handle_connect_smu()
            elif event == '-DISCONNECT_SMU-':   self._handle_disconnect_smu()

            # --- List Generation ---
            elif event == '-IV_FORM_CHK-':      self._handle_list_gen_form_check(values)
            elif event == '-VIZ_IV-':           self._handle_visualize_iv(values)
            elif event == '-SAVE_IV-':          self._handle_save_iv()
            elif event == '-VIZ_PULSE-':        self._handle_visualize_pulse(values)
            elif event == '-SAVE_PULSE-':       self._handle_save_pulse()

             # --- Manual Measurement ---
            elif event == '-RUN_MAN_DCIV-':     self._execute_manual_dciv(values)
            elif event == '-RUN_MAN_PULSE-':    self._execute_manual_pulse(values)

            # --- Protocol Builder ---
            elif event == '-PROTO_ADD-':        self._add_protocol_step(values)
            elif event == '-PROTO_REMOVE-':     self._remove_protocol_step()
            elif event == '-PROTO_CLEAR-':      self._clear_protocol()
            elif event == '-PROTO_RUN-':        self._execute_protocol(values)


        self.window.close()
        print("Standalone Measurement GUI closed.")


if __name__ == '__main__':
    app = StandaloneTesterGUI()
    app.run()