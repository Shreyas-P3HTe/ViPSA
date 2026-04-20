"""
This file contains the main class for the crossbar.
It contains the methods for connecting to the crossbar, switching channels, and measuring the channels.

"""

import inspect
from datetime import datetime

from vipsa.hardware.Multiplexing import Multiplexer
from vipsa.hardware.Source_Measure_Unit import KeysightSMU
from vipsa.analysis import Datahandling as dh #
import time 
import os
import pandas as pd #
import numpy as np #
from scipy.optimize import minimize

# Define a default save directory if needed, or get it from GUI/config
DEFAULT_SAVE_DIRECTORY = "./crossbar_data" # Example default

class Crossbar_Methods():

    def __init__(self, smu_resource_index=0, mux_port='COM3'): # Add initialization
        """Initializes the Crossbar_Methods class."""
        self.mux = None
        self.smu = None
        self.mux_port = mux_port #
        self.smu_resource_index = smu_resource_index
        self.data_handler = dh.Data_Handler() # Instantiate Data_Handler
        self.list_maker = dh.Listmaker() # Instantiate Listmaker
        self.is_smu_connected = False
        self.is_mux_connected = False

    @staticmethod
    def _supports_kwarg(callable_obj, kwarg_name):
        try:
            return kwarg_name in inspect.signature(callable_obj).parameters
        except (TypeError, ValueError):
            return False

    def _call_with_supported_kwargs(self, callable_obj, *args, **kwargs):
        supported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if self._supports_kwarg(callable_obj, key)
        }
        return callable_obj(*args, **supported_kwargs)

    def _build_measurement_metadata(self, data_name, device_id, measurement_params):
        smu_address = None
        if self.smu is not None and hasattr(self.smu, "get_address"):
            try:
                smu_address = self.smu.get_address()
            except Exception:
                smu_address = None
        return {
            "saved_at": datetime.now().isoformat(),
            "data_name": data_name,
            "sample_id": "crossbar",
            "device_id": device_id,
            "smu": {
                "class": type(self.smu).__name__ if self.smu is not None else None,
                "address": smu_address,
            },
            "measurement_parameters": measurement_params,
        }

# =============================================================================#
#                               Basic Methods                                  #
# =============================================================================#

    def connect_multiplexer(self):
        """Connects to the multiplexer."""
        print("Connecting to the multiplexer...")
        try:
            self.mux = Multiplexer() #
            self.mux.connect_mux() # Use the method from Multiplexing.py
            if self.mux.ser: # Check if connection was successful (ser object exists)
                self.is_mux_connected = True
                print("Multiplexer connected successfully.")
            else:
                self.is_mux_connected = False
                print("Failed to connect to Multiplexer.")
        except Exception as e:
            self.is_mux_connected = False
            print(f"Error connecting to Multiplexer: {e}")
            self.mux = None # Ensure mux is None if connection failed

    def disconnect_multiplexer(self):
        """Disconnects from the multiplexer."""
        if self.mux and self.is_mux_connected:
            try:
                self.mux.disconnect_mux() #
                self.is_mux_connected = False
                print("Multiplexer disconnected.")
            except Exception as e:
                 print(f"Error disconnecting Multiplexer: {e}")
        else:
            print("Multiplexer is not connected.")
        self.mux = None


    def connect_SMU(self):
        """Connects to the Keysight SMU."""
        print("Connecting to SMU...")
        try:
            self.smu = KeysightSMU(self.smu_resource_index) # Use the class from Source_Measure_Unit.py
            self.is_smu_connected = True
            print(f"SMU connected successfully at address: {self.smu.get_address()}") #
        except Exception as e:
            self.is_smu_connected = False
            print(f"Error connecting to SMU: {e}")
            self.smu = None # Ensure smu is None if connection failed

    def disconnect_SMU(self):
        """Disconnects from the SMU (Optional, as pyvisa usually handles closing)."""
        # pyvisa's resource manager typically handles closing connections when the object goes out of scope
        # or explicitly with rm.close(). Direct disconnect isn't usually needed for KeysightSMU class as written.
        if self.smu and self.is_smu_connected:
            print("SMU disconnected (handled by PyVISA resource manager).")
            self.is_smu_connected = False
        else:
             print("SMU is not connected.")
        self.smu = None


    def switch_channels(self, ch1, ch2):
        """Switches the multiplexer to the specified channels."""
        if not self.mux or not self.is_mux_connected:
            print("Error: Multiplexer not connected. Cannot switch channels.")
            return False

        print(f"Switching channels to CH1: {ch1}, CH2: {ch2}...")
        # Channel numbers are 1-16 in GUI/input, but Multiplexer class expects 0-15 index
        command_sent = self.mux.channel_to_command(ch1, ch2) #
        if command_sent:
            print("Channels switched successfully.")
            # Optional: Read response from Arduino
            # responses = self.mux.send_command("", self.mux.ser) # Send empty command to read buffer
            # if responses:
            #     print("Arduino Response:", responses)
            return True
        else:
            print("Failed to send channel switch command.")
            return False

    # --- Optimization methods are kept as they don't depend on stage/camera ---
    def quick_run_pulse(self, pulse_train_params, pulse_compliance=0.01): # Removed SMU=None, use self.smu
        """Helper function for pulse optimization."""
        if not self.smu or not self.is_smu_connected:
            print("Error: SMU not connected.")
            return np.inf # Return a large value indicating failure for minimization

        smu_address = self.smu.get_address() #

        vset, vreset, vread, width = pulse_train_params
        print(f"Optimizing Pulse - Testing Params: Vset={vset}, Vreset={vreset}, Vread={vread}, Width={width}")

        # Generate pulse list using Listmaker
        # Assuming 1 write, 1 erase, 1 read before, 1 read after per "cycle" for evaluation
        times, voltages = self.list_maker.generate_pulsing_data(
            write_pulses=1, write_voltage=vset, write_width=width,
            read_pulses=1, read_voltage=vread, read_width=width,
            erase_pulses=1, erase_voltage=vreset, erase_width=width,
            cycles=10 # Number of cycles for evaluation
        ) #

        # Run pulsed measurement using SMU class method
        pulsed_data = self.smu.pulsed_measurement(
            csv_path=None, # Provide list directly
            current_compliance=pulse_compliance,
            set_width=width,
            bare_list=voltages, # Pass the generated list
            adr=smu_address
        ) #

        if pulsed_data is None or len(pulsed_data) == 0:
            print("Warning: No data received from pulsed measurement during optimization.")
            return np.inf # Return large value if measurement failed

        # Analyze results using Data_Handler
        try:
            ionoff, consistency = self.data_handler.quick_pulse_analysis(pulsed_data) #
             # Define a performance metric to minimize (e.g., negative of Ion/Ioff, lower is better)
             # Or maximize (e.g. ionoff / consistency) - adjust `minimize` accordingly if maximizing
            if consistency == 0: consistency = 1e-9 # Avoid division by zero
            perf_metric = -ionoff # Example: Minimize the negative on/off ratio (maximize on/off)
            print(f"Optimization Step: Ion/Ioff={ionoff:.2f}, Consistency={consistency:.2f}, Metric={perf_metric:.2f}")

        except Exception as e:
             print(f"Warning: Error during pulse analysis: {e}")
             perf_metric = np.inf # Penalize errors

        return perf_metric

    def optimize_pulse_train(self, device_ID, save_directory, initial_params, pulse_compliance): # Removed SMU=None
        """Optimizes pulse parameters for a given device."""
        if not self.smu or not self.is_smu_connected:
            print("Error: SMU not connected. Cannot optimize pulse train.")
            return None
        if not self.mux or not self.is_mux_connected:
             print("Error: Multiplexer not connected. Cannot optimize pulse train.")
             return None


        print(f"Optimizing pulse train for Device ID: {device_ID}...")

        # Define bounds for optimization parameters (Vset, Vreset, Vread, Width) if necessary
        # Example bounds: bounds = [(0.1, 3.0), (-3.0, -0.1), (0.05, 0.5), (1e-5, 1e-3)]
        bounds = None # Adjust as needed

        try:
            result = minimize(
                fun=self.quick_run_pulse, # Pass the instance method
                x0=initial_params,
                args=(pulse_compliance,), # Pass additional fixed arguments
                method='L-BFGS-B', # Choose an appropriate method
                bounds=bounds, # Add bounds if defined
                options={'maxiter': 20, 'disp': True} # Adjust options
            )

            if result.success:
                best_params = result.x
                print(f"Optimization successful. Best parameters found: {best_params}")

                # Generate the final pulse sequence with optimized parameters
                vset_opt, vreset_opt, vread_opt, width_opt = best_params
                best_times, best_voltages = self.list_maker.generate_pulsing_data(
                     write_pulses=1, write_voltage=vset_opt, write_width=width_opt,
                     read_pulses=1, read_voltage=vread_opt, read_width=width_opt,
                     erase_pulses=1, erase_voltage=vreset_opt, erase_width=width_opt,
                     cycles=1000 # Generate a longer sequence for saving
                ) #

                # Save the optimized pulse list to a CSV
                filename = os.path.join(save_directory, f"device_{device_ID}_optimized_pulse.csv")
                os.makedirs(save_directory, exist_ok=True)
                self.list_maker.save_sweep_to_csv(best_times, best_voltages, filename) #
                print(f"Optimized pulse list saved to: {filename}")
                return filename
            else:
                print(f"Optimization failed: {result.message}")
                return None

        except Exception as e:
            print(f"An error occurred during optimization: {e}")
            return None

# =============================================================================#
#                               Compound Methods                               #
# =============================================================================#

    def run_single_DCIV(self, ch1, ch2, pos_compl, neg_compl, sweep_path,
                        sweep_delay=None, acq_delay =None, plot=True, save=True, save_dir=None,
                        use_4way_split=True, current_autorange=False, include_read_probe=True,
                        read_probe_mode="between_segments", progress_callback=None):
        """Runs a single DCIV sweep measurement on the specified channels."""
        if not self.smu or not self.is_smu_connected:
            print("Error: SMU not connected.")
            return None
        if not self.mux or not self.is_mux_connected:
            print("Error: Multiplexer not connected.")
            return None
        if not sweep_path or not os.path.exists(sweep_path):
             print(f"Error: Sweep file not found at {sweep_path}")
             return None


        print(f"Running single DCIV sweep for CH1: {ch1}, CH2: {ch2}")

        if not self.switch_channels(ch1, ch2):
            print("Failed to switch channels. Aborting measurement.")
            return None

        smu_address = self.smu.get_address() #
        try:
            resistance_data = None
            if use_4way_split and hasattr(self.smu, "list_IV_sweep_split_4"):
                sweep_data, resistance_data = self._call_with_supported_kwargs(
                    self.smu.list_IV_sweep_split_4,
                    sweep_path,
                    pos_compliance=pos_compl,
                    neg_compliance=neg_compl,
                    delay=sweep_delay,
                    acq_delay=acq_delay,
                    adr=smu_address,
                    include_read_probe=include_read_probe,
                    read_probe_mode=read_probe_mode,
                    current_autorange=current_autorange,
                    progress_callback=progress_callback,
                )
            else:
                sweep_result = self._call_with_supported_kwargs(
                    self.smu.list_IV_sweep_split,
                    sweep_path,
                    pos_compl,
                    neg_compl,
                    delay=sweep_delay,
                    acq_delay=acq_delay,
                    adr=smu_address,
                    include_read_probe=include_read_probe,
                    read_probe_mode=read_probe_mode,
                    current_autorange=current_autorange,
                    progress_callback=progress_callback,
                )
                if isinstance(sweep_result, tuple) and len(sweep_result) >= 2:
                    sweep_data, resistance_data = sweep_result[0], sweep_result[1]
                else:
                    sweep_data = sweep_result

            if sweep_data is not None and len(sweep_data) > 0:
                print("DCIV measurement successful.")
                if save:
                    if save_dir is None:
                        save_dir = DEFAULT_SAVE_DIRECTORY
                    device_id = f"{ch1}-{ch2}" # Create a unique ID
                    sweep_metadata = self._build_measurement_metadata(
                        "Sweep",
                        device_id,
                        {
                            "sweep_path": sweep_path,
                            "pos_compliance_a": pos_compl,
                            "neg_compliance_a": neg_compl,
                            "sweep_delay_s": sweep_delay,
                            "acquire_delay_s": acq_delay,
                            "use_4way_split": use_4way_split,
                            "include_read_probe": include_read_probe,
                            "read_probe_mode": read_probe_mode,
                            "current_autorange": current_autorange,
                            "mux_channels": {"ch1": ch1, "ch2": ch2},
                        },
                    )
                    # Contact current and Z_pos are not relevant here, save placeholder or 0
                    saved_file_sweep = self.data_handler.save_file(
                        sweep_data, "Sweep", "crossbar", device_id, 0, 0,
                        save_directory=save_dir, metadata=sweep_metadata
                    ) #
                    saved_file_resistance = None
                    if resistance_data is not None and len(resistance_data) > 0:
                        resistance_metadata = self._build_measurement_metadata(
                            "Resistance",
                            device_id,
                            {
                                "linked_sweep_path": sweep_path,
                                "include_read_probe": include_read_probe,
                                "read_probe_mode": read_probe_mode,
                                "mux_channels": {"ch1": ch1, "ch2": ch2},
                            },
                        )
                        saved_file_resistance = self.data_handler.save_file(
                            resistance_data, "Resistance", "crossbar", device_id, 0, 0,
                            save_directory=save_dir, metadata=resistance_metadata
                        )
                    
                    if plot and saved_file_sweep:
                         # Use methods from Datahandling.py
                        self.data_handler.show_plot(saved_file_sweep, "crossbar", device_id) #
                        if saved_file_resistance:
                             self.data_handler.show_resistance(saved_file_resistance, "crossbar", device_id) #

                return sweep_data, resistance_data, saved_file_sweep
            else:
                print("DCIV measurement failed or returned no data.")
                return None, None, None

        except Exception as e:
            print(f"Error during DCIV measurement: {e}")
            return None, None, None
        finally:
             # It's good practice to turn off output after measurement, though list_IV_sweep_split might handle this internally
             # self.smu.send_command(':OUTP1 OFF') # Turn off channel 1
             # self.smu.send_command(':OUTP2 OFF') # Turn off channel 2
             pass


    def run_single_pulse(self, ch1, ch2, compliance, pulse_path,
                         pulse_width=None, plot=True, save=True, save_dir=None,
                         set_acquire_delay=None, current_autorange=False):
        """Runs a single pulsed measurement on the specified channels."""
        if not self.smu or not self.is_smu_connected:
            print("Error: SMU not connected.")
            return None
        if not self.mux or not self.is_mux_connected:
            print("Error: Multiplexer not connected.")
            return None
        if not pulse_path or not os.path.exists(pulse_path):
             print(f"Error: Pulse file not found at {pulse_path}")
             return None


        print(f"Running single pulsed measurement for CH1: {ch1}, CH2: {ch2}")

        if not self.switch_channels(ch1, ch2):
            print("Failed to switch channels. Aborting measurement.")
            return None

        smu_address = self.smu.get_address() #
        try:
            # Use pulsed_measurement from Source_Measure_Unit.py
            pulse_data = self.smu.pulsed_measurement(
                csv_path=pulse_path,
                current_compliance=compliance,
                set_width=pulse_width, # Pass pulse_width if provided
                set_acquire_delay=set_acquire_delay,
                adr=smu_address,
                current_autorange=current_autorange,
            ) #

            if pulse_data is not None and len(pulse_data) > 0:
                print("Pulsed measurement successful.")
                if save:
                    if save_dir is None:
                        save_dir = DEFAULT_SAVE_DIRECTORY
                    device_id = f"{ch1}-{ch2}" # Create unique ID
                    pulse_metadata = self._build_measurement_metadata(
                        "Pulse",
                        device_id,
                        {
                            "pulse_path": pulse_path,
                            "compliance_a": compliance,
                            "pulse_width_s": pulse_width,
                            "set_acquire_delay_s": set_acquire_delay,
                            "current_autorange": current_autorange,
                            "mux_channels": {"ch1": ch1, "ch2": ch2},
                        },
                    )
                    # Contact current and Z_pos are not relevant here, save placeholder or 0
                    saved_file_pulse = self.data_handler.save_file(
                        pulse_data, "Pulse", "crossbar", device_id, 0, 0,
                        save_directory=save_dir, metadata=pulse_metadata
                    ) #

                    if plot and saved_file_pulse:
                         # Use method from Datahandling.py
                         self.data_handler.show_pulse(saved_file_pulse, "crossbar", device_id) #

                return pulse_data, saved_file_pulse
            else:
                print("Pulsed measurement failed or returned no data.")
                return None, None

        except Exception as e:
            print(f"Error during pulsed measurement: {e}")
            return None, None
        finally:
             # Turn off output after measurement
             # self.smu.send_command(':OUTP1 OFF') # Turn off channel 1
             # self.smu.send_command(':OUTP2 OFF') # Turn off channel 2
             pass


    def run_light_test(self, ch1, ch2, read_voltage, compliance, delay_s):
        """Measures current under illumination at intervals."""
        if not self.smu or not self.is_smu_connected:
            print("Error: SMU not connected.")
            return None
        if not self.mux or not self.is_mux_connected:
            print("Error: Multiplexer not connected.")
            return None
        # Add light controller connection check if you integrate one

        print(f"Running light test for CH1: {ch1}, CH2: {ch2}")

        if not self.switch_channels(ch1, ch2):
            print("Failed to switch channels. Aborting test.")
            return None

        smu_address = self.smu.get_address() #
        light_test_data = []
        try:
             # Configure SMU for DC measurement
            smu_inst = self.smu.rm.open_resource(smu_address) # Get pyvisa instrument
            smu_inst.write('*RST') #
            smu_inst.write('*CLS') #
            smu_inst.write(':SOUR1:FUNC:MODE VOLT') #
            smu_inst.write(f':SOUR1:VOLT {read_voltage}') #
            smu_inst.write(':SENS1:FUNC "CURR"') #
            smu_inst.write(f':SENS1:CURR:PROT {compliance}') #
            smu_inst.write(':SENS1:CURR:RANG:AUTO ON') # Or set a fixed range
            smu_inst.write(':OUTP1 ON') #
            smu_inst.query('*OPC?') # Wait for settings

             # --- Add Light Control Here ---
            print("Turning light ON (placeholder)")
            # e.g., light_controller.turn_on()

            start_time = time.time()
            while True: # Loop for measurement duration or external stop
                 current_time = time.time() - start_time
                 current = float(smu_inst.query(':MEAS:CURR?')) #
                 light_test_data.append((current_time, read_voltage, current))
                 print(f"Time: {current_time:.2f}s, Current: {current:.4e}A")

                 # --- Check for stop condition (e.g., GUI button, duration) ---
                 if current_time > 60: # Example: Stop after 60 seconds
                      break

                 time.sleep(delay_s)

            print("Turning light OFF (placeholder)")
            # e.g., light_controller.turn_off()

            smu_inst.write(':OUTP1 OFF') #
            smu_inst.close()

            # Save the data
            device_id = f"{ch1}-{ch2}"
            save_dir = DEFAULT_SAVE_DIRECTORY
            saved_file = self.data_handler.save_file(light_test_data, "LightTest", "crossbar", device_id, 0, 0, save_directory=save_dir) #
            print(f"Light test data saved to {saved_file}")

            return light_test_data

        except Exception as e:
            print(f"Error during light test: {e}")
            # Ensure output is off in case of error
            try:
                smu_inst.write(':OUTP1 OFF') #
                smu_inst.close()
            except: pass
            return None


    def measure_selected(self, selected_devs, protocol, save_dir=None):
        """Runs a defined protocol on a list of selected devices."""
        if not self.smu or not self.is_smu_connected:
            print("Error: SMU not connected. Cannot run protocol.")
            return
        if not self.mux or not self.is_mux_connected:
            print("Error: Multiplexer not connected. Cannot run protocol.")
            return

        if save_dir is None:
            save_dir = DEFAULT_SAVE_DIRECTORY

        print(f"Starting protocol measurement for {len(selected_devs)} devices.")

        for ch1, ch2 in selected_devs:
            print(f"\n--- Measuring Device CH1:{ch1}, CH2:{ch2} ---")
            device_save_dir = os.path.join(save_dir, f"device_{ch1}-{ch2}")
            os.makedirs(device_save_dir, exist_ok=True)

            for test_step in protocol:
                test_type = test_step.get('type') # Assume protocol is a list of dicts
                params = test_step.get('params', {})
                print(f"Executing Step: {test_type} with params: {params}")

                if test_type == 'DCIV':
                    self.run_single_DCIV(
                        ch1, ch2,
                        pos_compl=params.get('pos_compl', 0.001),
                        neg_compl=params.get('neg_compl', 0.01),
                        sweep_path=params.get('sweep_path'),
                        sweep_delay=params.get('sweep_delay'),
                        acq_delay=params.get('acq_delay'),
                        plot=params.get('plot', False), # Plotting might overwhelm if many devices
                        save=True,
                        save_dir=device_save_dir
                    )
                elif test_type == 'PULSE':
                     self.run_single_pulse(
                        ch1, ch2,
                        compliance=params.get('compliance', 0.01),
                        pulse_path=params.get('pulse_path'),
                        pulse_width=params.get('pulse_width'),
                        plot=params.get('plot', False),
                        save=True,
                        save_dir=device_save_dir
                     )
                elif test_type == 'OPTIMIZE_PULSE':
                     # Initial params might come from the protocol definition or a previous step
                     initial_params = params.get('initial_params', [1.0, -1.0, 0.1, 1e-4])
                     self.optimize_pulse_train(
                          device_ID=f"{ch1}-{ch2}",
                          save_directory=device_save_dir, # Save optimized pulse list here
                          initial_params=initial_params,
                          pulse_compliance=params.get('compliance', 0.01)
                     )
                elif test_type == 'LIGHT_TEST':
                     self.run_light_test(
                          ch1, ch2,
                          read_voltage=params.get('read_voltage', 0.1),
                          compliance=params.get('compliance', 1e-5),
                          delay_s=params.get('delay_s', 1.0)
                          # Save dir is handled within run_light_test
                     )

                # Add other test types as needed

                # Add a small delay between steps if necessary
                time.sleep(0.1)

            print(f"--- Finished Device CH1:{ch1}, CH2:{ch2} ---")
            # Add a delay between devices if needed
            time.sleep(0.5)

        print("\nProtocol measurement finished for all selected devices.")

# Example Usage (Optional - for testing within this file)
if __name__ == '__main__':
    crossbar_tester = Crossbar_Methods()

    # --- Connection Example ---
    crossbar_tester.connect_multiplexer()
    crossbar_tester.connect_SMU()

    if crossbar_tester.is_mux_connected and crossbar_tester.is_smu_connected:

        # --- Single Measurement Example ---
        print("\n--- Running Single DCIV Example ---")
        # Make sure sweep file exists at this path
        sweep_file = "C:/Users/amdm/Desktop/sweep patterns/Sweep_2Dmems_faster.csv" # Replace with your actual path
        if not os.path.exists(sweep_file):
            print(f"ERROR: Example sweep file not found: {sweep_file}. Skipping DCIV test.")
        else:
            crossbar_tester.run_single_DCIV(ch1=1, ch2=1, pos_compl=0.001, neg_compl=0.01, sweep_path=sweep_file, plot=False, save_dir="./example_output")

        print("\n--- Running Single Pulse Example ---")
         # Make sure pulse file exists at this path
        pulse_file = "C:/Users/amdm/Desktop/sweep patterns/Pulses_1000.csv" # Replace with your actual path
        if not os.path.exists(pulse_file):
             print(f"ERROR: Example pulse file not found: {pulse_file}. Skipping Pulse test.")
        else:
             crossbar_tester.run_single_pulse(ch1=1, ch2=2, compliance=0.005, pulse_path=pulse_file, plot=False, save_dir="./example_output")


        # --- Protocol Measurement Example ---
        print("\n--- Running Protocol Example ---")
        selected_devices = [(1, 1), (1, 2), (2, 1), (2,2)] # Example device list (ch1, ch2)
        # Define a protocol as a list of dictionaries
        measurement_protocol = [
            {'type': 'DCIV', 'params': {'sweep_path': sweep_file, 'pos_compl': 0.001, 'neg_compl': 0.005, 'plot': False}},
            {'type': 'PULSE', 'params': {'pulse_path': pulse_file, 'compliance': 0.005, 'pulse_width': 1e-4, 'plot': False}}
            # Add other steps like {'type': 'OPTIMIZE_PULSE', 'params': {...}} or {'type': 'LIGHT_TEST', 'params': {...}}
        ]
        # Ensure files exist before running protocol that uses them
        if os.path.exists(sweep_file) and os.path.exists(pulse_file):
             crossbar_tester.measure_selected(selected_devices, measurement_protocol, save_dir="./example_protocol_output")
        else:
             print("Skipping protocol example because sweep or pulse file not found.")


    # --- Disconnection Example ---
    crossbar_tester.disconnect_multiplexer()
    crossbar_tester.disconnect_SMU() # Optional
