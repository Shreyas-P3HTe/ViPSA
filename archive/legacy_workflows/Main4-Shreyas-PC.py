# -*- coding: utf-8 -*-

"""
Created on Thu Jan  9 14:12:13 2025

@author: Shreyas
"""

import os
import numpy as np
import pandas as pd
import time
import random
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from datetime import datetime

from Source_Measure_Unit import KeysightSMU, pyvisa, KeithleySMU
from Openflexture import Light, stage, Zaber
from Vision import get_contours, get_contour_distances, capture_image
import Datahandling as dh

class Vipsa_Methods():
    
    def connect_equipment(self, SMU_name):
        
        '''
        In case you want to use an internal method without connecting to the equipment externally,
        use this function to connect to all
        
        Args : 
            None
        Returns : 
            Bool of whether the equipment is connected
        '''
        try :
            self.top_light = Light()
            self.top_light.control_lights("rainbow")            
            self.stage = stage('COM5', 115200, 1)
            self.Zaber = Zaber('COM7')
            self.zaber_x, self.zaber_y = self.Zaber.get_devices()
            self.SMU_name = SMU_name
            if SMU_name == "KeysightB2901BL":
                self.SMU = KeysightSMU(0)
            elif SMU_name == "Keithley2450":
                self.SMU = KeithleySMU(0)
                
            self.equipment = True
            
            self.sweep_path = "C:/Users/amdm/Desktop/sweep patterns/Sweep_2Dmems.csv"
            self.pulse_path = "C:/Users/amdm/Desktop/sweep patterns/Pulse20.csv"
            
            for i in range (1,3):
                self.top_light.control_lights("green")            
                self.top_light.control_lights("off")  
                
        except Exception as e :
            
            print("Error occured while connecting : ", e)
            
            self.equipment = False
            if not self.top_light :
                self.top_light = Light()
            self.top_light.control_lights("red") 
            time.sleep(3)
            self.top_light.control_lights("off")
        
        return self.equipment
    
    def disconnect_equipment(self):

        try:        
            if self.stage:
                self.stage.move_z_by(-30)
                self.stage.disconnect()
                                
            if self.Zaber:
                self.zaber_x.device.connection.close()
                self.zaber_y.device.connection.close()
                print("Zaber devices disconnected")
            
            if self.top_light :
                self.top_light.disconnect()
                
            self.equipment = False
            
        except Exception as e :
            
            print ("Error occured while disconnecting : ", e)
            
            return self.equipment
            
# =============================================================================#
#                               Basic Methods                                  #
# =============================================================================#

    
    def detect_contact_and_move_z(self, SMU=None, stage=None, 
                                  step_size=1, test_voltage=0.1, 
                                  lower_threshold=1e-9, upper_threshold=5e-8,
                                  max_attempts=50, delay=0.05):
        '''
        Moves the arduino stage 1 step at a time and checks if the current is 
        between the lower and upper threshold. Stops when the current is withtin the desired range
        
        Args : 
            step_size : int, float
                Step size for height increment
            test_voltage : float
                Voltage for checking contact. MUST BE small enough to not affect the device performance
                Recommended : <10-15% of switching voltage
            lower_threshold & upper_threshold : float
                threshold for contact current
            max_attempts : int
                how many times should we move up before giving up ?
            delay : float
                time delay to allow system to settle down
                
        Returns : 
            Tuple with all the connections to be used internally
        '''
        if SMU is None:
            SMU = self.SMU
        if stage is None:
            stage = self.stage
    
        current = 0
        contact_detected = False
    
        try:
            # configure once before the loop, not every time
            self.SMU.write("*RST")
            self.SMU.write(":SOUR:FUNC VOLT")
            self.SMU.write(":SENS:FUNC 'CURR'")
            self.SMU.write(f":SOUR:VOLT:LEV {test_voltage}")
            if self.SMU_name == "Keithley2450":
                self.SMU.write(":ROUT:TERM REAR")
                print("Active terminals:", (self.SMU.ask(":ROUT:TERM?")).strip())
            self.SMU.write(f":SENS:CURR:PROT {10*upper_threshold}")
            self.SMU.write(":OUTP ON")
            # Try to probe, pray that you get it correct in the first go
            for attempt in range(max_attempts):
                current = SMU.get_contact_current_fast(test_voltage, settle=delay)
                print(f"Attempt {attempt+1}: Current = {current:.3e} A")
    
                if current > lower_threshold:  # check if the current exceeds
                    contact_detected = True
                    break
    
                #if attempt %5 == 0:
                    #stage.move_z_by(2.5*step_size)
                #else:
                stage.move_z_by(step_size)
    
            if contact_detected and current > upper_threshold:
    
                # Oh golly heck, you went too far ! Now slowly retrace your steps
                backstep = 0
                while current > upper_threshold:
                    stage.move_z_by(-0.5 * step_size)
                    backstep += 1
                    time.sleep(delay)
                    current = SMU.get_contact_current_fast(test_voltage, settle=delay)
                    print(f"Retracing steps : Current = {current:.3e} A")
    
                    if backstep % 10 == 0:
                        stage.move_z_by(-2.5 * step_size)
    
                time.sleep(delay * 5)
                current = SMU.get_contact_current_fast(test_voltage, settle=delay)
                print(f"Stabilized,  Current = {current:.3e} A") 
    
                # Too much. TOO MUCH ! UGH !
                i = 0
                while current < lower_threshold:
                    stage.move_z_by(0.25 * step_size)
                    current = SMU.get_contact_current_fast(test_voltage, settle=delay)
                    print(f"Final attempt {i}: Current = {current:.3e} A")
                    i += 1
                    if i > 20:
                        break
    
                if current > upper_threshold:
                    stage.move_z_by(-0.1 * step_size)
    
            # Alas ! You didn't pray hard enough.
            if not contact_detected:
                print("Could not establish electrical contact.")
    
            contact_height = stage.get_current_position()[2]
            return contact_detected, current, contact_height
    
        except Exception as e:
            print("Error occured while connecting :", e)
            return False, 0.0, None
    
        finally:
            try:
                self.SMU.write(":OUTP OFF")
            except Exception:
                pass

            
    def center_pad(self, x_distances, y_distances, 
                   stage = None):
        
        '''
        Centers the pad using the Openflexture stage. Kinda redundant function.
        The correction is retained for subsequent devices.
        '''
        
        # NOTE : the value of 1.55 and its sign (-ve, here) are a virtue of 
        #       layout of optics and stages, and MUST NOT BE CHANGED
        
        if stage == None:
            stage = self.stage
            
        stage.move_xy_by((1.55 * x_distances[0]), (1.55* y_distances[0]))
                
    def center_pad_zaber(self, x_distance, y_distance, 
                         Zaber_x = None, Zaber_y = None):
        
        '''
        Centers the pad using Zaber stages.
        Faster than the Openflexture and also no problem of runnning out of limits.
        
        Only drawback is that for large arrays that are significantly tilted,
        the correction resets for every device.
        '''
        
        # NOTE : the value of 25 and its sign (+ve, here) are a virtue of 
        #       layout of optics and stages, and MUST NOT BE CHANGED
        
        if Zaber_x == None and Zaber_y == None :
            Zaber_x = self.zaber_x
            Zaber_y = self.zaber_y
        
        x = float(-x_distance[0]*30)
        y = float(-y_distance[0]*30)     
        Zaber_x.move_relative(x)
        Zaber_y.move_relative(y)
        
    def correct_course(self, move=True, zaber_corr=True, recheck =True,
                       Zaber_x = None, Zaber_y = None,
                       stage = None, top_light = None):
        
        '''
        
        Clicks a photo (or two) and corrects the misalignment.
        
        Args :
            self,
            move : You can turn off correction movement. 
                Effectively renders this function as a mere photopoint
            zaber_corr : When True, uses center_pad_zaber(). Else uses center_pad()
            recheck : When True, clicks and shows another photo post-correction.
                Turn this off for samples highly sensitive to light exposure.
            
        returns :
            None
        
        '''
        if stage == None:
            stage = self.stage
        
        if Zaber_x == None and Zaber_y == None :
            Zaber_x = self.zaber_x
            Zaber_y = self.zaber_y
        
        if top_light == None :
            top_light = self.top_light
        
        im_no = 2 # no. of images plotted after correction. 
        
        top_light.control_lights('on') # Say cheese !        
        image_array = capture_image(0)
        
        while image_array is None:
            print("Capture failed, trying again.")
            image_array = capture_image(0)
        top_light.control_lights('off')
        
        contours, _ = get_contours(image_array)
        x_distances, y_distances, cont_image = get_contour_distances(image_array, contours)
        
        if move :
            
            if zaber_corr :
                self.center_pad_zaber(x_distances, y_distances, Zaber_x, Zaber_y)
            else :
                self.center_pad(x_distances, y_distances, stage)
        
        if recheck : #this checks if the correction has taken place.
            top_light.control_lights('on') # Say cheese again !
            image_corrected = capture_image(0)
            while image_corrected is None:
                print("Capture failed, trying again.")
                image_corrected = capture_image(0)
            top_light.control_lights('off')
            im_no = 3
        
        # Visualization
        fig, axarr = plt.subplots(1, im_no, figsize=(5*im_no, 5), subplot_kw={'xticks': [], 'yticks': []})
        center_y, center_x = image_array.shape[0] // 2, image_array.shape[1] // 2
        axarr[0].imshow(image_array, cmap='gray')
        axarr[0].set_title('Before correction')
        axarr[0].plot(center_x, center_y, 'rx')
        axarr[1].imshow(cont_image, cmap='gray')
        axarr[1].set_title('Contour Image')
        axarr[1].plot(center_x, center_y, 'rx')
        
        if recheck :
            axarr[2].imshow(image_corrected, cmap='gray')
            axarr[2].set_title('After correction')
            axarr[2].plot(center_x, center_y, 'rx')
            
        plt.tight_layout()
        plt.show()
    
    def quick_run_pulse(self, pulse_train_params, pulse_compliance=0.01, SMU=None):
        
        if SMU == None:
            SMU = self.SMU
            
        SMU_adress = SMU.get_address()        
        listmaker = dh.Listmaker()
        data_handler = dh.Data_Handler()
        
        vset, vreset, vread, width = pulse_train_params
        print(pulse_train_params)
        times, voltages = listmaker.generate_pulsing_data(1, vset, width, 
                                                     1, vreset, width, 
                                                     1, vread, width, 
                                                     10)
        
        pulsed_data = SMU.pulsed_measurement(csv_path = None, current_compliance = pulse_compliance,
                                             set_width = width, adr = SMU_adress, bare_list = voltages )
        
        #print(pulsed_data)
        
        results = data_handler.quick_pulse_analysis(pulsed_data)
        
        ionoff, consistency = results
        perf_metric = 0.5*ionoff + 0.5*consistency
        
        return perf_metric
        
    def optimize_pulse_train(self, device_ID, save_directory, initial_params, pulse_compliance, SMU=None):
        
        if SMU == None:
            SMU = self.SMU
            
        SMU_adress = SMU.get_address() 
        listmaker = dh.Listmaker()
        data_handler = dh.Data_Handler()
        
        best_recipe = minimize(
        		fun = self.quick_run_pulse,
        		x0 = initial_params,
        		method='L-BFGS-B',
                	options={'maxiter': 10})
        
        best_times, best_voltages = listmaker.generate_pulsing_data(1, best_recipe[0], best_recipe[3],
                                                      1, best_recipe[1], best_recipe[3],
                                                      1, best_recipe[2], best_recipe[3],
                                                      1000)
        
        filename = f"{save_directory}/{device_ID}_best_p"
        
        best_recipe_file = listmaker.save_sweep_to_csv(best_times, best_voltages, filename)
        
        return best_recipe_file

    def extrapolate_z_positions(self, grid_size, z_positions):
        
        """
        Extrapolates Z positions for a 5x5 grid based on corner Z positions.
    
        Parameters:
            grid_size (tuple): The size of the grid (rows, columns).
            z_positions (dict): Corner Z positions as {device: z_value}.
    
        Returns:
            np.ndarray: Extrapolated Z positions for the entire grid.
        """
        rows, cols = grid_size
        grid = np.zeros((rows, cols))
    
        # Assign corner Z positions to the grid
        grid[0, 0] = z_positions[1]   # Top-left (Device 1)
        grid[0, -1] = z_positions[5]  # Top-right (Device 5)
        grid[-1, 0] = z_positions[21] # Bottom-left (Device 21)
        grid[-1, -1] = z_positions[25] # Bottom-right (Device 25)
    
        # Perform bilinear interpolation for the rest of the grid
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0:  # Skip the corners
                    top_interp = grid[0, 0] + c * (grid[0, -1] - grid[0, 0]) / (cols - 1)
                    bottom_interp = grid[-1, 0] + c * (grid[-1, -1] - grid[-1, 0]) / (cols - 1)
                    grid[r, c] = top_interp + r * (bottom_interp - top_interp) / (rows - 1)

        return grid

    def get_grid_z_positions(self, file_path):
        
        df = pd.read_csv(file_path)
        
        # Validate the "Device" column
        if "Device" not in df.columns:
            raise ValueError("The CSV file must contain a 'Device' column.")
        
        devices = df["Device"].tolist()
        
        # Define the corner devices for a 5x5 grid
        corner_devices = [1, 5, 21, 25]
        z_positions = {}
        
        # Measure Z positions for corner devices
        for device in corner_devices:
            
            
            if device in devices:
                is_contact, contact_current, z_pos = self.detect_contact_and_move_z(device)
                
                if is_contact:
                    z_positions[device] = z_pos
                else:
                    raise RuntimeError(f"No contact detected for Device {device}.")
            else:
                raise ValueError(f"Device {device} is not listed in the CSV.")
        
        # Extrapolate Z positions for the entire grid
        grid_size = (5, 5)  # 5x5 grid
        z_grid = self.extrapolate_z_positions(grid_size, z_positions)
        
        # Output results
        # Flatten the Z grid and add it to the DataFrame
        df["Z_Position"] = z_grid.flatten()
    
        # Save the updated DataFrame back to the file
        df.to_csv(file_path, index=False)
    
        # Output results
        print("Extrapolated Z positions for the grid:")
        print(z_grid)

    #def measure_resistance(self, read_V)

# =============================================================================#
#                               Compund Methods                                #
# =============================================================================#

    def run_single_DCIV(self, sample_no, device_no, pos_compl, neg_compl, sweep_delay, step_no=None, acq_delay =None,
                        plot = True, align = True, approach = True, zaber_corr = True, corr_recheck = True, #for correct_course
                        step_size=0.5, test_voltage=0.05, lower_threshold=1e-11, upper_threshold=5e-11, max_attempts=50, delay=1, #for detect_contact_and_move_z
                        save_directory = None, sweep_path = None, wait_time = 0,
                        SMU = None, stage = None, Zaber_x = None, Zaber_y = None, top_light = None,
                        compliance_pf=None, compliance_pb=None, compliance_nf=None, compliance_nb=None,
                        use_4way_split=True,):
        """
        Runs a DCIV and saves the file.
        Replaces the old "measure_and_save" method, does exactly the same.
        
        
        Args :
        ----------
            sample_no : Sample ID
            device_no : Device ID
            plot : TYPE, optional
                Do you want to see the plot?. The default is True.
            align : TYPE, optional
                Is the device misaligned and needs correction ?. The default is True.
            approach : TYPE, optional
                Is the contact not yet made and you want to auto-approach ?. The default is True.


        Returns :
        -------
            Bool 'is_measured' for whether the measurement was complete & device height (needed for retracting)

        """
        if SMU == None :
            SMU = self.SMU

        if save_directory == None:
            save_directory = self.save_directory
        
        if stage == None:
            stage = self.stage
        
        if Zaber_x == None and Zaber_y == None :
            Zaber_x = self.zaber_x
            Zaber_y = self.zaber_y
        
        if top_light == None :
            top_light = self.top_light
            
        Data_Handler = dh.Data_Handler() 
        SMU_adress = SMU.get_address()
        
        if align : #misaligned device
            self.correct_course(move = True, zaber_corr = zaber_corr, recheck = corr_recheck,
                                Zaber_x = Zaber_x, Zaber_y = Zaber_y, stage = stage, top_light = top_light)
            
        if approach : 
            contact, cont_current, height = self.detect_contact_and_move_z(SMU = SMU, stage = stage,
                                                                           step_size = step_size, test_voltage = test_voltage, 
                                                                           lower_threshold = lower_threshold, upper_threshold = upper_threshold, 
                                                                           max_attempts = max_attempts , delay = delay )
            
        else :
            contact = True
            cont_current = SMU.get_contact_current(test_voltage,adr=SMU_adress)
            height = stage.get_current_position()[2]
        
        if contact :
            
            # ---- derive 4-way compliances if not provided ----
            # Default behavior:
            #   + polarity uses pos_compl
            #   - polarity uses neg_compl
            # You can override any of these four individually.
            if compliance_pf is None: compliance_pf = pos_compl
            if compliance_pb is None: compliance_pb = pos_compl
            if compliance_nf is None: compliance_nf = neg_compl
            if compliance_nb is None: compliance_nb = neg_compl
    
            if use_4way_split:
                sweep_data, resistance_data = SMU.list_IV_sweep_split_4(
                    sweep_path,
                    compliance_pf=compliance_pf,
                    compliance_pb=compliance_pb,
                    compliance_nf=compliance_nf,
                    compliance_nb=compliance_nb,
                    delay=sweep_delay,
                    # you can pass nplc / wait_time too if your keithley method supports it
                    wait_time=wait_time,
                )
            else:
                # fallback to existing 2-way
                sweep_data, resistance_data = SMU.list_IV_sweep_split(
                    sweep_path, pos_compl, neg_compl,
                    delay=sweep_delay, acq_delay=acq_delay, adr=SMU_adress,
                    wait_time=wait_time
                )
            
            print(f"wait time between measurements: {wait_time}s")
            #time.sleep(wait_time)
            
            saved_file_s = Data_Handler.save_file(sweep_data, "Sweep", sample_no, device_no, step_no, cont_current, height, save_directory)
            saved_resistance = Data_Handler.save_file(resistance_data, "Resistance", sample_no, device_no, step_no, cont_current, height, save_directory)
            is_measured = True
            if plot :
                
                Data_Handler.show_plot(saved_file_s, sample_no, device_no)
                Data_Handler.show_resistance(saved_resistance, sample_no, device_no)
            
        elif not contact : 
            is_measured = False
            height = stage.get_current_position()[2]
            saved_file_s = None
            
            
        return is_measured, height, saved_file_s
            
    def run_resistance_measurement(self, sample_no, device_no, pos_compl, neg_compl, sweep_delay, step_no=None, acq_delay =None,
                        plot = True, align = True, approach = True, zaber_corr = True, corr_recheck = True, #for correct_course
                        step_size=0.5, test_voltage=0.1, lower_threshold=1e-11, upper_threshold=5e-11, max_attempts=50, delay=1, #for detect_contact_and_move_z
                        save_directory = None, sweep_path = None, 
                        SMU = None, stage = None, Zaber_x = None, Zaber_y = None, top_light = None):
        """
        TINY resistance measurement
        
        
        Args :
        ----------
            sample_no : Sample ID
            device_no : Device ID
            plot : TYPE, optional
                Do you want to see the plot?. The default is True.
            align : TYPE, optional
                Is the device misaligned and needs correction ?. The default is True.
            approach : TYPE, optional
                Is the contact not yet made and you want to auto-approach ?. The default is True.


        Returns :
        -------
            Bool 'is_measured' for whether the measurement was complete & device height (needed for retracting)

        """
        if SMU == None :
            SMU = self.SMU

        if save_directory == None:
            save_directory = self.save_directory
        
        if stage == None:
            stage = self.stage
        
        if Zaber_x == None and Zaber_y == None :
            Zaber_x = self.zaber_x
            Zaber_y = self.zaber_y
        
        if top_light == None :
            top_light = self.top_light
            
        Data_Handler = dh.Data_Handler() 
        SMU_adress = SMU.get_address()
        
        if align : #misaligned device
            self.correct_course(move = True, zaber_corr = zaber_corr, recheck = corr_recheck,
                                Zaber_x = Zaber_x, Zaber_y = Zaber_y, stage = stage, top_light = top_light)
            
        if approach : 
            contact, cont_current, height = self.detect_contact_and_move_z(SMU = SMU, stage = stage,
                                                                           step_size = step_size, test_voltage = test_voltage, 
                                                                           lower_threshold = lower_threshold, upper_threshold = upper_threshold, 
                                                                           max_attempts = max_attempts , delay = delay )
            
        else :
            contact = True
            cont_current = SMU.get_contact_current(test_voltage,adr=SMU_adress)
            height = stage.get_current_position()[2]
        
        if contact :
            
            #sweep_data = SMU.list_IV_sweep_manual(sweep_path, pos_compl, neg_compl, delay=sweep_delay, adr=SMU_adress)
            sweep_data = SMU.list_IV_sweep_split(sweep_path, pos_compl, neg_compl, delay=sweep_delay, acq_delay=acq_delay, adr=SMU_adress)
            saved_file_s = Data_Handler.save_file(sweep_data, "Sweep", sample_no, device_no, step_no, cont_current, height, save_directory)
            is_measured = True
            if plot :
                
                Data_Handler.show_resistance(saved_file_s, sample_no, device_no)
        
        elif not contact : 
            is_measured = False
            height = stage.get_current_position()[2]
            saved_file_s = None
            
            
        return is_measured, height, saved_file_s 

    
    def run_single_pulse(self, sample_no, device_no, compliance, pulse_width, step_no=None,
                        plot = True, align = True, approach = True, zaber_corr = True, corr_recheck = True, #for correct_course
                        step_size=0.5, test_voltage=0.1, lower_threshold=1e-11, upper_threshold=5e-11, max_attempts=50, delay=1, #for detect_contact_and_move_z
                        save_directory = None, pulse_path = None, 
                        SMU = None, stage = None, Zaber_x = None, Zaber_y = None, top_light = None):
        """
        Runs a pulse measurement and saves the file.
        Replaces the old "measure_and_save" method, does exactly the same.
        Use this for running device endurance test
        
        Args :
        ----------
            sample_no : Sample ID
            device_no : Device ID
            plot : TYPE, optional
                Do you want to see the plot?. The default is True.
            align : TYPE, optional
                Is the device misaligned and needs correction ?. The default is True.
            approach : TYPE, optional
                Is the contact not yet made and you want to auto-approach ?. The default is True.
        Returns :
        -------
            Bool 'is_measured' for whether the measurement was complete & device height (needed for retracting)

        """
        if SMU == None :
            SMU = self.SMU

        if save_directory == None:
            save_directory = self.save_directory
        
        if stage == None:
            stage = self.stage
        
        if Zaber_x == None and Zaber_y == None :
            Zaber_x = self.zaber_x
            Zaber_y = self.zaber_y
        
        if top_light == None :
            top_light = self.top_light
            
        
        SMU_adress = SMU.get_address()
        Data_Handler = dh.Data_Handler()
        
        if align : #misaligned device
            self.correct_course(move = True, zaber_corr = zaber_corr, recheck = corr_recheck,
                                Zaber_x = Zaber_x, Zaber_y = Zaber_y, stage = stage, top_light = top_light)
            
        if approach : 
            contact, cont_current, height = self.detect_contact_and_move_z(SMU = SMU, stage = stage,
                                                                           step_size = step_size, test_voltage = test_voltage, 
                                                                           lower_threshold = lower_threshold, upper_threshold = upper_threshold, 
                                                                           max_attempts = max_attempts , delay = delay )
            
        else :
            contact = True
            cont_current = SMU.get_contact_current(test_voltage, adr=SMU_adress)
            height = stage.get_current_position()[2]
        
        if contact :
            
            pulse_data = SMU.pulsed_measurement(pulse_path, compliance, pulse_width, adr = SMU_adress)
            #pulse_data = SMU.list_IV_sweep_manual(pulse_path, compliance, compliance*100, delay=pulse_width, adr=SMU_adress)
            saved_file_p = Data_Handler.save_file(pulse_data, "Pulse", sample_no, device_no, step_no, cont_current, height, save_directory)
            is_measured = True
            if plot :
                
                Data_Handler.show_pulse(saved_file_p, sample_no, device_no)
        
        elif not contact : 
            is_measured = False
            height = stage.get_current_position()[2]
            saved_file_p = None
            
        return is_measured, height, saved_file_p
            
    def measure_IV_gridwise(self, sample_ID, gridpath, pos_compl, neg_compl, sweep_delay, acq_delay = None,
                        skip_instances = 1, startpoint = 1, randomize = False,
                        plot = True, align = True, approach = True, zaber_corr = True, corr_recheck = True, #for correct_course
                        step_size=0.5, test_voltage=0.25, lower_threshold=1e-11, upper_threshold=5e-11, max_attempts=100, delay=1, #for detect_contact_and_move_z
                        save_directory = None, sweep_path = None, pulse_path = None, update_sweep = False, wait_time = 0,
                        SMU = None, stage = None, Zaber_x = None, Zaber_y = None, top_light = None,
                        compliance_pf = None, compliance_pb = None, compliance_nf = None, compliance_nb = None,
                        use_4way_split = True):

        """
        Performs IV measurements on a grid of devices.

        This function iterates through a grid of device coordinates, moves the stage to each 
        position, and executes an IV measurement at each location. 
                
        Args :
        ----------
            sample_ID : sample name    
        
            plot : TYPE, optional
                Do you want to see the plot?. The default is True.
            align : TYPE, optional
                Is the device misaligned and needs correction ?. The default is True.
            approach : TYPE, optional
                Is the contact not yet made and you want to auto-approach ?. The default is True.


        Returns :
        -------
            Bool 'is_measured' for whether the measurement was complete & device height (needed for retracting)

        """
        
        if SMU == None :
            SMU = self.SMU

        if save_directory == None:
            save_directory = self.save_directory
        
        if stage == None:
            stage = self.stage
        
        if Zaber_x == None and Zaber_y == None :
            Zaber_x = self.zaber_x
            Zaber_y = self.zaber_y
        
        if top_light == None :
            top_light = self.top_light
            
            
        #get metadata from user
        
        # sthumi = input("Enter Humidity :")
        # sttemp = input("Enter Temperature : ")
        # sttime = datetime.now()
            
        
        grid = np.genfromtxt(gridpath, delimiter=',', skip_header=1)        
        
        # Skip instances and optionally randomize the points
        grid_to_move = grid[::skip_instances]
        
        if randomize:
            np.random.seed(42)  # Set a seed for reproducibility if needed
            grid_to_move = np.random.permutation(grid_to_move)
        
        print(grid_to_move)
        
        # meas_seq = [1, 2, 3,
        #             6, 5, 4,
        #             7, 8, 9]
        
        meas_seq = [ 1, 2, 3, 4,
                     8, 7, 6, 5,
                    9 ,10, 11, 12, 
                    16, 15, 14, 13]
        #             

        # meas_seq = [ 1, 2, 3, 4, 5,
        #             10, 9, 8, 7, 6,
        #             11,12,13,14,15,
        #             20,19,18,17,16,
        #             21,22,23,24,25]
                    
        # meas_seq = [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,
        #             20,19,18,17,16,15,14,13,12,11,
        #             21,22,23,24,25,26,27,28,29,30,
        #             40,39,38,37,36,35,34,33,32,31,
        #             41,42,43,44,45,46,47,48,49,50,
        #             60,59,58,57,56,55,54,53,52,51,
        #             61,62,63,64,65,66,67,68,69,70,
        #             80,79,78,77,76,75,74,73,72,71,
        #             81,82,83,84,85,86,87,88,89,90,
        #            100,99,98,97,96,95,94,93,92,91]
        
        # Create a dictionary to map device IDs to their grid coordinates
        device_to_grid = {device_ID: grid_point for grid_point in grid_to_move for device_ID in grid_point[:1]}
        
        # Sort grid_to_move based on meas_seq
        grid_to_move_sorted = sorted(grid_to_move, key=lambda x: meas_seq.index(x[0])) 
        
        start_index = meas_seq.index(startpoint)
        i = start_index        
        # Move the stage gridwise and perform measurement
        for point in grid_to_move_sorted[start_index:]:
            
            device_ID, x_coord, y_coord = point
            print(f"Moving to device {device_ID} at X: {x_coord}, Y: {y_coord}")
            # ... (Rest of your existing code within the for loop) ...
            
            # Move the stage
            Zaber_x.move_absolute(x_coord)
            Zaber_y.move_absolute(y_coord)
            
            
            stage.move_z_by(30)
            
            # Measure and save the data
            meas_status, z_height, file = self.run_single_DCIV(sample_ID, device_ID, pos_compl, neg_compl, sweep_delay, acq_delay,
                                plot, align, approach, zaber_corr, corr_recheck,                                #for correct_course
                                step_size, test_voltage, lower_threshold, upper_threshold, max_attempts, delay, #for detect_contact_and_move_z
                                save_directory, sweep_path, wait_time,
                                SMU, stage, Zaber_x, Zaber_y, top_light,
                                compliance_pf, compliance_pb, compliance_nf, compliance_nb,
                                use_4way_split)
            
            stage.flush()
            
            if i<24 and meas_seq[i] > meas_seq[i+1] :
                stage.move_z_by(-50) #coming back
                
            else :
                stage.move_z_by(-50) #going ahead
            
            i = i+1
        
        print("Completed testing gridwise.")
        
        # #get metadata
        # entime = datetime.now()
        # entemp = input("Enter Temperature : ")
        # enhumi = input("Enter Humidity : ")
        
        
        # dh.Data_Handler.save_metadata(self, sample_ID, sttime, entime, sthumi, sttemp, enhumi, entemp, 
        #                               protocol='Grid IV sweep with dual sweep & resistance measurement',
        #                               save_directory = save_directory)
                    
    def adaptive_testing(self, gridpath, sample_ID, pos_compl, neg_compl, pulse_compliance, sweep_delay, pulse_width, 
                        skip_instances = 1, startpoint = 0, randomize = False,
                        plot = True, align = True, approach = True, zaber_corr = True, corr_recheck = True, #for correct_course
                        step_size=0.5, test_voltage=0.1, lower_threshold=1e-11, upper_threshold=5e-11, max_attempts=100, delay=1, #for detect_contact_and_move_z
                        save_directory = None, sweep_path = None, pulse_path = None, update_sweep = False,
                        SMU = None, stage = None, Zaber_x = None, Zaber_y = None, top_light = None):
       
        if SMU == None :
            SMU = self.SMU

        if save_directory == None:
            save_directory = self.save_directory
        
        if stage == None:
            stage = self.stage
        
        if Zaber_x == None and Zaber_y == None :
            Zaber_x = self.zaber_x
            Zaber_y = self.zaber_y
        
        if top_light == None :
            top_light = self.top_light
            
        
        SMU_adress = SMU.get_address()
        listmaker = dh.Listmaker()
        data_handler = dh.Data_Handler()
        
                
        if update_sweep :
        #measure sacrificial device
        
            device_ID = -1
            
            stage.move_z_by(-5)
            
            meas_status, z_height, file = self.run_single_DCIV(sample_ID, device_ID, pos_compl, neg_compl, sweep_delay, 
                                plot, align, approach, zaber_corr, corr_recheck, #for correct_course
                                step_size, test_voltage, lower_threshold, upper_threshold, max_attempts, delay, #for detect_contact_and_move_z
                                save_directory, sweep_path, 
                                SMU, stage, Zaber_x, Zaber_y, top_light)
        
            
            data = data_handler.IV_calculations(file)
            
            if data and len(data) == 4:
                # Extract the maximum value from each sublist
                max_values = [max(sublist) if sublist else None for sublist in data]
                Vset_max, on_off_max, Vread_max, Vcomp_max = max_values
            
            
            # Generate an adjusted sweep_csv for subsequent measurements
            
            sweep_path = listmaker.make_updated_sweep(sweep_path, Vcomp_max, -1*Vcomp_max, 0.02, 0.001)
            
            stage.move_z_by(-110)
        
        #Gridwise measurement
        
        grid = np.genfromtxt(gridpath, delimiter=',', skip_header=1)        
        
        # Skip instances and optionally randomize the points
        grid_to_move = grid[::skip_instances]
        
        if randomize:
            np.random.seed(42)  # Set a seed for reproducibility if needed
            grid_to_move = np.random.permutation(grid_to_move)
        
        print(grid_to_move)
        
        meas_seq = [1,2,3,4,5,10,9,8,7,6,11,12,13,14,15,20,19,18,17,16,21,22,23,24,25]
        
        # Create a dictionary to map device IDs to their grid coordinates
        device_to_grid = {device_ID: grid_point for grid_point in grid_to_move for device_ID in grid_point[:1]}
        
        # Sort grid_to_move based on meas_seq
        grid_to_move_sorted = sorted(grid_to_move, key=lambda x: meas_seq.index(x[0])) 
        
        start_index = meas_seq.index(startpoint)
        i = start_index        
        # Move the stage gridwise and perform measurement
        for point in grid_to_move_sorted[start_index:]:
            
            device_ID, x_coord, y_coord = point
            print(f"Moving to device {device_ID} at X: {x_coord}, Y: {y_coord}")
            # ... (Rest of your existing code within the for loop) ...
            
            # Move the stage
            Zaber_x.move_absolute(x_coord)
            Zaber_y.move_absolute(y_coord)
            
            
            stage.move_z_by(20)
            
            # Measure and save the data
            meas_status, z_height, file = self.run_single_DCIV(sample_ID, device_ID, pos_compl, neg_compl, sweep_delay, 
                                plot, align, approach, zaber_corr, corr_recheck,                                #for correct_course
                                step_size, test_voltage, lower_threshold, upper_threshold, max_attempts, delay, #for detect_contact_and_move_z
                                save_directory, sweep_path, 
                                SMU, stage, Zaber_x, Zaber_y, top_light)
            
            print("Running calculations...")
            data = data_handler.IV_calculations(file)                       
            
            if data[0] == None:
                print("The device doesn't seem healthy, moving on")
                stage.flush()
                continue
            
            if data and len(data) == 4:
                # Extract the maximum value from each sublist
                max_values = [max(sublist) if sublist else None for sublist in data]
                Vset_max, on_off_max, Vread_max, Vcomp_max = max_values
                
            if on_off_max > 10 :
                
                Vread = 0.1*Vset_max
                
                print("The device is working properly\nMoving to Endurance testing...")
                
                if pulse_path is None:  # Generate pulse list by itself
                    # Construct the file path
                    pulse_path = f"{save_directory}/pulse_list/{device_ID}.csv"
                    
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(pulse_path), exist_ok=True)
                    
                    # Check if the file exists; if not, create it
                    if not os.path.isfile(pulse_path):
                        # Create an empty DataFrame (or populate it with default columns if needed)
                        empty_df = pd.DataFrame(columns=["Column1", "Column2"])  # Replace with your desired columns
                        empty_df.to_csv(pulse_path, index=False)
                        print(f"File {pulse_path} created.")
                    else:
                        print(f"File {pulse_path} already exists.")
                    
                    times, voltages = listmaker.generate_pulsing_data(1, 1.5*Vcomp_max, pulse_width, #set
                                                    1, Vread, pulse_width, #read
                                                    1, -2*Vcomp_max, pulse_width, #reset
                                                    1000)
                    
                    listmaker.save_sweep_to_csv(times, voltages, pulse_path)
                
                align_bef_pulse =0
                approach_bef_pulse =0
                
                self.run_single_pulse(sample_ID, device_ID, pulse_compliance, pulse_width, 
                                    plot, align_bef_pulse, approach_bef_pulse, zaber_corr, corr_recheck,                                #for correct_course
                                    step_size, test_voltage, lower_threshold, upper_threshold, max_attempts, delay,            #for detect_contact_and_move_z
                                    save_directory, pulse_path, 
                                    SMU, stage, Zaber_x, Zaber_y, top_light)  
                                    
            stage.flush()
            
            if meas_seq[i] > meas_seq[i+1]:
                stage.move_z_by(-60)
                
            else :
                stage.move_z_by(-30)
            
            i = i+1
        print("Completed testing gridwise.")

    def best_5_endurance(self, gridpath, sample_ID, pos_compl, neg_compl, pulse_compliance, sweep_delay, pulse_width, 
                        skip_instances = 1, startpoint = 0, randomize = False,
                        plot = True, align = True, approach = True, zaber_corr = True, corr_recheck = True, #for correct_course
                        step_size=0.5, test_voltage=0.1, lower_threshold=1e-11, upper_threshold=5e-11, max_attempts=100, delay=1, #for detect_contact_and_move_z
                        save_directory = None, sweep_path = None, pulse_path = None, update_sweep = False,
                        SMU = None, stage = None, Zaber_x = None, Zaber_y = None, top_light = None):
        
        #init#
        if SMU == None :
            SMU = self.SMU

        if save_directory == None:
            save_directory = self.save_directory
        
        if stage == None:
            stage = self.stage
        
        if Zaber_x == None and Zaber_y == None :
            Zaber_x = self.zaber_x
            Zaber_y = self.zaber_y
        
        if top_light == None :
            top_light = self.top_light
            
        SMU_adress = SMU.get_address()
        listmaker = dh.Listmaker()
        data_handler = dh.Data_Handler()
        
        
        self.measure_IV_gridwise(self, sample_ID, gridpath, pos_compl, neg_compl, sweep_delay, 
                            skip_instances, startpoint, randomize,
                            plot, align, approach, zaber_corr, corr_recheck, #for correct_course
                            step_size, test_voltage, lower_threshold, upper_threshold, max_attempts, delay, #for detect_contact_and_move_z
                            save_directory, sweep_path, pulse_path, update_sweep,
                            SMU, stage, Zaber_x, Zaber_y, top_light)
        
        #get a list of top 5 devices
        
        #go to X,Y
        
        
        self.correct_course(move = True, zaber_corr = zaber_corr, recheck = corr_recheck,
                            Zaber_x = Zaber_x, Zaber_y = Zaber_y, stage = stage, top_light = top_light)
        
        #go to z-5 steps
        
        contact, height, cont_current = self.detect_contact_and_move_z(SMU = SMU, stage = stage,
                                                                       step_size = step_size, test_voltage = test_voltage, 
                                                                       lower_threshold = lower_threshold, upper_threshold = upper_threshold, 
                                                                       max_attempts = max_attempts , delay = delay )
        
        
# =============================================================================#        
                 #Protocol backend: save/load/validate/execute ---
# =============================================================================#

    def save_protocol(self, filepath, protocol_list_configs):
        """Save a protocol (list of step dicts) to JSON file."""
        import json
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(protocol_list_configs, f, indent=2)
            print(f"Protocol saved to {filepath}")
            return True
        except Exception as e:
            print("Error saving protocol:", e)
            return False

    def load_protocol(self, filepath):
        """Load a protocol JSON file and return list of configs."""
        import json
        try:
            with open(filepath, 'r') as f:
                protocol = json.load(f)
            print(f"Protocol loaded from {filepath}")
            return protocol
        except Exception as e:
            print("Error loading protocol:", e)
            return None

    def validate_protocol(self, protocol_list_configs):
        """Basic validation: ensure list of dicts with 'type' and 'params'."""
        if not isinstance(protocol_list_configs, list):
            return False, "Protocol must be a list"
        for i,step in enumerate(protocol_list_configs):
            if not isinstance(step, dict):
                return False, f"Step {i} is not a dict"
            if 'type' not in step or 'params' not in step:
                return False, f"Step {i} missing 'type' or 'params'"
        return True, "OK"

    def _execute_protocol_step(self, config, sample_no=None, device_no=None, step_no = None, save_directory=None,
                               SMU=None, stage=None, Zaber_x=None, Zaber_y=None, top_light=None):
        """Execute a single protocol step config. Returns a result dict."""
        ctype = config.get('type')
        params = config.get('params', {})
        result = {'type': ctype, 'params': params, 'status': 'failed', 'output': None}

        # Defaults
        if SMU is None:
            SMU = getattr(self, 'SMU', None)
        if stage is None:
            stage = getattr(self, 'stage', None)
        if Zaber_x is None or Zaber_y is None:
            z = getattr(self, 'zaber_x', None)
            y = getattr(self, 'zaber_y', None)
            if z is not None and y is not None:
                Zaber_x = z
                Zaber_y = y
        if top_light is None:
            top_light = getattr(self, 'top_light', None)

        selected_smu = SMU
        if isinstance(SMU, dict):
            default_smu = 'keithley' if ctype == 'DCIV' else 'keysight'
            requested_smu = str(params.get('smu', params.get('smu_type', default_smu))).strip().lower()
            selected_smu = SMU.get(requested_smu)
            if selected_smu is None:
                available = ", ".join(sorted(SMU.keys()))
                raise ValueError(f"Requested SMU '{requested_smu}' is not connected. Available: {available}")

        try:
            if ctype == 'DCIV':
                pos_compl = float(params.get('pos_compl', params.get('pos_comp', 1e-3)))
                neg_compl = float(params.get('neg_compl', params.get('neg_comp', -1e-3)))
                sweep_delay = params.get('sweep_delay', None)
                sweep_path = params.get('sweep_path', None)
                align = params.get('align', True)
                approach = params.get('approach', True)
                # call existing method
                out = self.run_single_DCIV(
                    sample_no=sample_no,
                    device_no=device_no,
                    pos_compl=pos_compl,
                    neg_compl=neg_compl,
                    sweep_delay=sweep_delay,
                    step_no=step_no,
                    acq_delay=None,
                    plot=False,
                    align=align,
                    approach=approach,
                    save_directory=save_directory,
                    sweep_path=sweep_path,
                    SMU=selected_smu,
                    stage=stage,
                    Zaber_x=Zaber_x,
                    Zaber_y=Zaber_y,
                    top_light=top_light,
                )
                result['status'] = 'ok'
                result['output'] = out

            elif ctype == 'PULSE':
                compliance = float(params.get('compliance', params.get('pulse_comp', 1e-3)))
                pulse_width = params.get('pulse_width', None)
                pulse_path = params.get('pulse_path', None)
                align = params.get('align', True)
                approach = params.get('approach', True)
                out = self.run_single_pulse(
                    sample_no=sample_no,
                    device_no=device_no,
                    compliance=compliance,
                    pulse_width=pulse_width,
                    step_no=step_no,
                    plot=False,
                    align=align,
                    approach=approach,
                    save_directory=save_directory,
                    pulse_path=pulse_path,
                    SMU=selected_smu,
                    stage=stage,
                    Zaber_x=Zaber_x,
                    Zaber_y=Zaber_y,
                    top_light=top_light,
                )
                result['status'] = 'ok'
                result['output'] = out

            elif ctype == 'ALIGN':
                # params may include zaber_corr and recheck
                zaber_corr = params.get('zaber_corr', True)
                recheck = params.get('recheck', True)
                self.correct_course(move=True, zaber_corr=zaber_corr, recheck=recheck,
                                     Zaber_x=Zaber_x, Zaber_y=Zaber_y, stage=stage, top_light=top_light)
                result['status'] = 'ok'
                result['output'] = None

            elif ctype == 'APPROACH':
                step_size = params.get('step_size', 1)
                test_voltage = params.get('test_voltage', 0.1)
                lower = params.get('lower_threshold', 1e-11)
                upper = params.get('upper_threshold', 5e-11)
                max_attempts = params.get('max_attempts', 50)
                delay = params.get('delay', 0.05)
                out = self.detect_contact_and_move_z(SMU=selected_smu, stage=stage, step_size=step_size,
                                                     test_voltage=test_voltage, lower_threshold=lower,
                                                     upper_threshold=upper, max_attempts=max_attempts, delay=delay)
                result['status'] = 'ok'
                result['output'] = out

            else:
                print(f"Unknown protocol step type: {ctype}")
                result['status'] = 'unknown'

        except Exception as e:
            print(f"Error executing protocol step {ctype}:", e)
            result['status'] = 'error'
            result['output'] = str(e)

        return result

    def run_protocol(self, protocol_list_configs, sample_no=None, device_no=None, save_directory=None,
                     SMU=None, stage=None, Zaber_x=None, Zaber_y=None, top_light=None, stop_on_error=False):
        """Execute a list of protocol step configs on a single target.

        Returns a list of results for each step.
        """
        ok, msg = self.validate_protocol(protocol_list_configs)
        if not ok:
            print("Invalid protocol:", msg)
            return None

        results = []
        for i, step in enumerate(protocol_list_configs):
            print(f"Executing protocol step {i+1}/{len(protocol_list_configs)}: {step.get('type')}")
            res = self._execute_protocol_step(step, sample_no=sample_no, device_no=device_no, step_no = i,
                                             save_directory=save_directory, SMU=SMU, stage=stage,
                                             Zaber_x=Zaber_x, Zaber_y=Zaber_y, top_light=top_light)
            results.append(res)
            if res.get('status') in ('error', 'failed') and stop_on_error:
                print(f"Stopping protocol due to error at step {i+1}")
                break

        print("Protocol execution complete.")
        return results



