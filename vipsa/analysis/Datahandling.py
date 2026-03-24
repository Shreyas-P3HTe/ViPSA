# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:29:28 2025

@author: shrey
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import colorcet as cc
import statistics as stat
import time

class Data_Handler():
    
    def __init_(self):
        self.save_directory = "C:/Users/amdm/OneDrive - Nanyang Technological University/ViPSA data folder/Spinbot_memristors_batch_1/gold"
    
    def save_metadata(self, sample_id, start_time, end_time,
                      start_hum, start_temp, end_hum, end_temp,
                      protocol=None, save_directory=None):
        """
        Saves measurement metadata for a measurement.
        """
        if save_directory is None:
            save_directory = self.save_directory
    
        # Create the directory if it doesn't exist
        sample_dir = f"{save_directory}/{sample_id}"
        os.makedirs(sample_dir, exist_ok=True)
    
        file_path = os.path.join(sample_dir, "metadata.txt")
        
        with open(file_path, 'a') as file:
            file.write(f"Measurement Start Time: {start_time}\n")
            file.write(f"Measurement End Time: {end_time}\n")
            file.write(f"Starting Humidity: {start_hum}\n")
            file.write(f"Ending Humidity: {end_hum}\n")
            file.write(f"Starting Temperature: {start_temp}\n")
            file.write(f"Ending Temperature: {end_temp}\n")
            
            if protocol:
                file.write(f"Protocol: {protocol}\n")
    
        print(f"Metadata saved to {file_path}")
    
    def save_file(self, data, data_name, sample_id, device_id, cont_current, Z_pos, step_no = None,
                  save_directory = None):
        '''

        Args :
            data : Dataframe needed to be saved - 3 columns in the order ['Time(T)', 'Voltage (V)', 'Current (A)']
            data_name : Whether Sweep or pulse
            sample_id : String, sample ID
            device_id : String, device ID
            cont_current : float, contact current (preferably from detect_contact_and_move_z)
            Z_pos : float, contact height (preferably from detect_contact_and_move_z)
            save_directory : TYPE, optional
                Save directory. The default is None.

        Returns :
            file_path : string, path of the CSV

        '''
            
        if save_directory == None:
            save_directory = self.save_directory            
        
        directory_path = f"{save_directory}/{sample_id}/{data_name}"
        os.makedirs(directory_path, exist_ok=True)
        if step_no == None :
            file_path = os.path.join(directory_path, f"device_{device_id}.csv")
        else :
            file_path = os.path.join(directory_path, f"device_{device_id}_step_{step_no}.csv")
        df = pd.DataFrame(data, columns=['Time(T)', 'Voltage (V)', 'Current (A)'])
        df['Contact Current (A)'] = cont_current
        df['Z position'] = Z_pos
        df.to_csv(file_path, index=True)
        print(f"Data saved to {file_path}")
        
        return file_path
    
    def show_plot(self, csvpath, sample_id = "NA", device_id = "NA"):
        
        '''
        Plots DC-IV from any csv on a semi-log (y log) scale.
        
        Needs to have following columns : 
            'Current (A)'
            'Voltage (V)'
            'Contact Current (A)'
            'Z position'
            
        Args :
            
            csvpath : CSV to be plotted. 
            
        '''
        
        df = pd.read_csv(csvpath)
        
        # Convert negative current values to their absolute values
        df['Current (A)'] = df['Current (A)'].abs()
        
        # Initialize a figure
        plt.figure(figsize=(10, 4), dpi=150)        
                
        
        num_colors = 25  # Adjust to match the number of cycles you want to plot
        
        # Generate a list of colors from the colormap
        colors = cc.glasbey[:num_colors]
        start_index = 0
        color_idx = 0  # Initialize color index
        
        legend_labels = []
        
        # Iterate through the DataFrame to find cycles
        for i in range(1, len(df)):
            # Detect when a full cycle is complete
            if df['Voltage (V)'].iloc[i-1] == 0 and df['Voltage (V)'].iloc[i] > 0:
                # This marks the end of a cycle and the start of a new one
                if start_index != i-1:  # Prevent plotting on first start
                    # Plot the completed cycle
                    plt.plot(df['Voltage (V)'].iloc[start_index:i], 
                             df['Current (A)'].iloc[start_index:i], 
                             linestyle='-', 
                             color=colors[color_idx % len(colors)], 
                             label=f'Cycle {color_idx + 1}')
                    # Append the cycle label to legend labels list
                    legend_labels.append(f'Cycle {color_idx + 1}')
                    # Change the color for the next cycle
                    color_idx += 1
                # Update the start index for the new cycle
                start_index = i
        
        # Plot the remaining cycle, if any
        if start_index < len(df):
            plt.plot(df['Voltage (V)'].iloc[start_index:], 
                     df['Current (A)'].iloc[start_index:], 
                     linestyle='-', 
                     color=colors[color_idx % len(colors)], 
                     label=f'Cycle {color_idx + 1}')
            legend_labels.append(f'Cycle {color_idx + 1}')
        
        Curr = df["Contact Current (A)"].iloc[1]
        Height = df["Z position"].iloc[1]
        
        # Set axis labels
        plt.ylabel('Current (log scale)')
        plt.xlabel('Voltage')
        
        # Set the y-axis to logarithmic scale
        plt.yscale('log')
        #plt.ylim(10e-9, 10e-2)        
        # Add grid
        plt.grid(True)
        
        # Add legend
        num_column = -(-num_colors // 15)
        plt.legend(title="Cycles", bbox_to_anchor = (1.02,1), loc= "upper left", ncol = num_column)
        plt.tight_layout()
        # Show the plot
        plt.title(f"Sample {sample_id} | Device {device_id} | Contact_current = {Curr} A | Contact_height = {Height}")
        plt.show() 
        
    def show_plot_with_dashes(self, csvpath, sample_id="NA", device_id="NA"):
    
        df = pd.read_csv(csvpath)
        df['Current (A)'] = df['Current (A)'].abs()
    
        plt.figure()
        colormap = cm.get_cmap('Set1')
        num_colors = 11
        colors = [colormap(i / num_colors) for i in range(num_colors)]
    
        def split_by_4_indices(vlist):
            """
            Splits the voltage sweep into pf/pb/nf/nb segments using indices.
            Returns a list of (tag, start_idx, end_idx)
            """
            splits = []
            current_tag = ""
            start_idx = 0
            pmax = max(vlist)
            nmax = min(vlist)
    
            for i in range(1, len(vlist)):
                v_prev = vlist[i - 1]
                v_now = vlist[i]
    
                if current_tag == "":
                    if v_now > 0:
                        current_tag = "pf"
                    elif v_now < 0:
                        current_tag = "nf"
                    start_idx = i - 1
    
                # pf to pb
                if current_tag == "pf" and v_now < pmax and v_now < v_prev:
                    splits.append((current_tag, start_idx, i))
                    current_tag = "pb"
                    start_idx = i - 1
    
                # pb to nf
                elif current_tag == "pb" and v_now < 0 and v_now < v_prev:
                    splits.append((current_tag, start_idx, i))
                    current_tag = "nf"
                    start_idx = i - 1
    
                # nf to nb
                elif current_tag == "nf" and v_now > nmax and v_now > v_prev:
                    splits.append((current_tag, start_idx, i))
                    current_tag = "nb"
                    start_idx = i - 1
    
                # nb to pf (start of new cycle)
                elif current_tag == "nb" and v_now > 0 and v_now > v_prev:
                    splits.append((current_tag, start_idx, i))
                    current_tag = "pf"
                    start_idx = i - 1
    
            # Add the final segment
            if current_tag and start_idx < len(vlist) - 1:
                splits.append((current_tag, start_idx, len(vlist)))
    
            return splits
    
        start_index = 0
        color_idx = 0
    
        for i in range(1, len(df)):
            # Detect start of new cycle
            if df['Voltage (V)'].iloc[i - 1] == 0 and df['Voltage (V)'].iloc[i] > 0:
                if start_index != i - 1:
                    subdf = df.iloc[start_index:i].reset_index(drop=True)
                    vlist = subdf['Voltage (V)'].tolist()
                    ilist = subdf['Current (A)'].tolist()
    
                    sweep_segments = split_by_4_indices(vlist)
    
                    for tag, s_idx, e_idx in sweep_segments:
                        linestyle = '-' if tag in ['pf', 'nf'] else '--'
                        plt.plot(vlist[s_idx:e_idx],
                                 ilist[s_idx:e_idx],
                                 linestyle=linestyle,
                                 color=colors[color_idx % len(colors)])
                    color_idx += 1
                start_index = i
    
        # Handle last cycle
        if start_index < len(df):
            subdf = df.iloc[start_index:].reset_index(drop=True)
            vlist = subdf['Voltage (V)'].tolist()
            ilist = subdf['Current (A)'].tolist()
    
            sweep_segments = split_by_4_indices(vlist)
    
            for tag, s_idx, e_idx in sweep_segments:
                linestyle = '-' if tag in ['pf', 'nf'] else '--'
                plt.plot(vlist[s_idx:e_idx],
                         ilist[s_idx:e_idx],
                         linestyle=linestyle,
                         color=colors[color_idx % len(colors)])
    
        # Final annotations
        Curr = df["Contact Current (A)"].iloc[1]
        Height = df["Z position"].iloc[1]
        plt.ylabel('Current (log scale)')
        plt.xlabel('Voltage')
        plt.yscale('log')
        plt.grid(True)
        plt.legend(title="Cycles")
        plt.tight_layout()
        plt.title(f"Sample {sample_id} | Device {device_id} | "
                  f"Contact_current = {Curr:.2e} A | Contact_height = {Height}")
        plt.show()

        
    def show_resistance(self, csvpath, sample_id = "NA", device_id = "NA", cycles = 11):

        df = pd.read_csv(csvpath)
        res_arr = []
        cycles_arr = []
        cycle_no = 0
        step_size = len(df) // 24
        print(step_size)
                
        for i in range(0, 2*cycles):
            start_idx = (i * 400)
            end_idx = (i + 1)*400
            res = df.iloc[start_idx:end_idx, 2] / df.iloc[start_idx:end_idx, 3]
            cycles_arr.append(cycle_no)
            
            if (i+1)%2 == 0:
                cycle_no +=1
                
            
            res_arr.append(abs(stat.mean(res)))
            print(abs(stat.mean(res))*1e-3,"kΩ")
            
        print(cycles_arr)
        
        for i in range (0,2*cycles):
            if i%2 == 0:
                plt.scatter(x= cycles_arr[i], y =res_arr[i], marker='.', c='red')
            else:
                plt.scatter(x= cycles_arr[i], y =res_arr[i], marker='.', c='green')
        
        plt.xlabel("Cycle number - 0th is forming")
        plt.ylabel("Resistance (kΩ, log scale)")
        legend = ["Off","On"]
        plt.legend(legend)
        plt.yscale("log")
        plt.show()
        
    #def show_resistance_single_point(self, csvpath,)
        
    def analyze_resistance_cycles(self, csvpath, num_cycles):
        """
        Analyzes resistance data from a CSV file over multiple cycles and plots the results.
    
        Args:
            csvpath (str): The path to the CSV file.
            num_cycles (int): The total number of cycles to analyze.
        """
        try:
            df = pd.read_csv(csvpath)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csvpath}")
            return
    
        if df.empty:
            print("Error: The CSV file is empty.")
            return
    
        # Define a meaningful constant for the chunk size
        CHUNK_SIZE = 400
        res_arr = []
        cycles_arr = []
        cycle_no = 0
    
        # Calculate step_size based on the DataFrame length
        step_size = len(df) // 24
        print(f"Calculated step size: {step_size}")
    
        # Iterate through the specified number of cycles (each cycle has 2 states: ON and OFF)
        for i in range(2 * num_cycles):
            start_idx = i * CHUNK_SIZE
            end_idx = (i + 1) * CHUNK_SIZE
    
            # Handle cases where the end index exceeds the DataFrame length
            if start_idx >= len(df):
                print(f"Warning: Start index {start_idx} exceeds DataFrame length. Stopping at cycle {cycle_no}.")
                break
            if end_idx > len(df):
                print(f"Warning: End index {end_idx} exceeds DataFrame length. Using data up to the end.")
                end_idx = len(df)
    
            # Extract the relevant data slice and calculate the resistance ratio
            try:
                resistance_ratio = df.iloc[start_idx:end_idx, 2] / df.iloc[start_idx:end_idx, 3]
            except ZeroDivisionError:
                print(f"Warning: Division by zero encountered in cycle iteration {i}. Skipping this segment.")
                continue
    
            # Calculate the absolute mean resistance for this segment
            mean_resistance = abs(stat.mean(resistance_ratio))
            res_arr.append(mean_resistance)
            cycles_arr.append(cycle_no)
    
            # Increment the cycle number after every two iterations (representing ON and OFF states)
            if (i + 1) % 2 == 0:
                cycle_no += 1
    
            print(f"Cycle {cycle_no-1 if (i+1)%2 == 0 else cycle_no}, Step {i%2}: {mean_resistance * 1e-3:.3f} kΩ")
    
        print("Cycle Numbers Array:", cycles_arr)
    
        # Plotting the resistance values against cycle numbers
        plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
    
        for i in range(len(cycles_arr)):
            if i % 2 == 0:
                plt.scatter(x=cycles_arr[i], y=res_arr[i], marker='.', c='red', label='Off' if i == 0 else "")
            else:
                plt.scatter(x=cycles_arr[i], y=res_arr[i], marker='.', c='green', label='On' if i == 1 else "")
    
        plt.xlabel("Cycle number - 0th is forming")
        plt.ylabel("Resistance (kΩ, log scale)")
        plt.yscale("log")
        plt.title("Resistance Variation Over Cycles")
        plt.legend(title="State")  # Add a title to the legend
        plt.grid(True, which="both", ls="-", alpha=0.5)  # Add a grid for better readability
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        plt.show()

    def show_pulse(self, csvpath, sample_id="NA", device_id="NA"):
        """
        Plots DC-IV from a CSV on a semi-log (y log) scale, coloring first and second reads differently.
        
        Args:
            csvpath: Path to the CSV file to be plotted.
            sample_id: Sample ID (optional).
            device_id: Device ID (optional).
        """
        
        df = pd.read_csv(csvpath)
        
        # Add the "Pulse number" column
        df["Pulse number"] = df.index // 8 + 1
        
        # Initialize a plot
        plt.figure(figsize=(20, 6))
        
        set_voltage = df["Voltage (V)"].max()
        reset_voltage = df["Voltage (V)"].min()
        
        # Filter for non-zero values between set and reset voltages
        read_voltage_candidates = df[
            (df["Voltage (V)"] > reset_voltage)
            & (df["Voltage (V)"] < set_voltage)
            & (df["Voltage (V)"] != 0)
        ]["Voltage (V)"]
        
        # Choose a read voltage (e.g., the first valid candidate)
        read_voltage = read_voltage_candidates.iloc[0] if not read_voltage_candidates.empty else None
        
        # Define colors for different voltage conditions
        colors = {"set": "green", "reset": "black"}
        if read_voltage is not None:
          colors["read1"] = "black"
          colors["read2"] = "red"  # Second read in blue
        
        # Plot based on voltage conditions
        for voltage, color in colors.items():
          if voltage in ["set", "reset"]:
            subset = df[df["Voltage (V)"] == voltage]
            plt.scatter(
                subset["Pulse number"], subset["Current (A)"], color=color, label=f"{voltage} Voltage", s=2
            )
          elif voltage == "read1":
            # Filter for first read voltage
            first_read_indices = df[df["Voltage (V)"] == read_voltage].index[:4]
            first_read_subset = df.loc[first_read_indices]
            plt.scatter(
                first_read_subset["Pulse number"],
                first_read_subset["Current (A)"],
                color=color,
                label=f"1st Read ({read_voltage} V)",
                s=2,
            )
          elif voltage == "read2":
            # Filter for second read voltage
            second_read_indices = df[df["Voltage (V)"] == read_voltage].index[4:]
            second_read_subset = df.loc[second_read_indices]
            plt.scatter(
                second_read_subset["Pulse number"],
                second_read_subset["Current (A)"],
                color=color,
                label=f"2nd Read ({read_voltage} V)",
                s=2,
            )
        
        # Set labels, legend, and title
        plt.xlabel("Pulse number")
        plt.ylabel("Current (A)")
        plt.title("Voltage vs Current by Pulse Number")
        plt.yscale("log")
        plt.legend()
        plt.grid()
        
        # Show the plot
        plt.show()
            
    def fold_pos_cycle(self, cycle_df):
        
        '''
        Folds the positive half cycle for Vset detection
        '''
        
        pos_max_idx = cycle_df['Voltage (V)'].idxmax()  # Index of positive max voltage
        folded_data = []  # List to store folded data
        for i in range(pos_max_idx + 1):  # Iterate from 0 to positive max index
            voltage = cycle_df['Voltage (V)'].iloc[i]
            current_fwd = cycle_df['Current (A)'].iloc[i]
            current_bck = cycle_df['Current (A)'].iloc[2*pos_max_idx +1 - i]
            on_off_ratio = current_bck/current_fwd
            folded_data.append((voltage, current_fwd, current_bck, on_off_ratio))
        
        # Convert folded data into a DataFrame
        folded_df = pd.DataFrame(folded_data, columns=['Voltage (V)', 'Current Forward (A)', 'Current Backward (A)', 'On Off Ratio'])
        return folded_df
                
    def IV_calculations(self, csvpath):

        """
        Perform IV calculations on the provided CSV file.

        This function processes the IV data contained in the CSV file to extract key metrics such as 
        threshold voltage (Vth), maximum On-Off ratio, voltage at maximum On-Off ratio (Vr), 
        and voltage at compliance point (Vcomp). It operates on specific cycles within the data and 
        computes these metrics for each selected cycle.

        Parameters:
        ----------
        csvpath : str
            The file path of the CSV file containing IV data. The CSV is expected to have columns 
            'Voltage (V)' and 'Current Forward (A)' starting from the third column onward.

        Returns:
        -------
        list
            A list containing four lists:
                - Vset_cycle: Threshold voltages (Vth) for the selected cycles.
                - on_off_cycle: Maximum On-Off ratios for the selected cycles.
                - Vread_cycle: Voltages (Vr) at maximum On-Off ratio for the selected cycles.
                - Vcomp_cycle: Voltages (Vcomp) at compliance points for the selected cycles.

        Notes:
        -----
        - Cycles are detected as segments of data where the voltage transitions from 0 to a positive value.
        - The function only processes cycles within the specified `cycle_range`.
        - Logarithmic and derivative calculations are performed on the forward current data to determine Vth.
        - On-Off ratio and compliance voltage are calculated based on predefined thresholds.

        Raises:
        ------
        Exception
            If an error occurs during processing, the function prints an error message and skips the file.
        """
        
        
        # Read the CSV file
        df = pd.read_csv(csvpath)
        df = df.iloc[:, 2:4].copy()
        
        # Initialize variables
        start_index = 0
        # Specify the range of cycles to process (e.g., 2 to 5)
        cycle_range = range(2, 6)
        cycles = []  # List to store individual cycles
    
        Vset_cycle = []
        on_off_cycle = []
        Vread_cycle = []
        Vcomp_cycle = [] #voltages at which device compliance takes place
    
        try :
            # Detect cycles and split the DataFrame
            for j in range(1, len(df)):
                # Check if a cycle starts
                if df['Voltage (V)'].iloc[j - 1] == 0 and df['Voltage (V)'].iloc[j] > 0:
                    # If not the first cycle, extract the cycle and append it to the list
                    if start_index != j - 1:
                        cycle_data = df.iloc[start_index:j].reset_index(drop=True)
                        cycles.append(cycle_data)
                    # Update the start index for the new cycle
                    start_index = j
        
            # Capture the last cycle if it exists
            if start_index < len(df):
                cycle_data = df.iloc[start_index:].reset_index(drop=True)
                cycles.append(cycle_data)
        
            # Process selected cycles
            for cycle_idx in cycle_range:
                if cycle_idx - 1 < len(cycles):
                    folded = self.fold_pos_cycle(cycles[cycle_idx - 1])
                    
                    # Compute derivatives
                    folded['Log10(Current Forward)'] = np.log10(folded['Current Forward (A)'])
                    folded['d(Log10(Current Forward))/dV'] = np.gradient(folded['Log10(Current Forward)'], folded['Voltage (V)'])
                    folded['d2(Log10(Current Forward))/dV2'] = np.gradient(folded['d(Log10(Current Forward))/dV'], folded['Voltage (V)'])
                    
                    # Determine Vth
                    Vth_index = folded['d(Log10(Current Forward))/dV'].idxmax()
                    Vth = folded['Voltage (V)'].iloc[Vth_index]
                    print()
                    # Determine max On-Off ratio and corresponding voltage
                    Vr_idx = folded['On Off Ratio'].idxmax()
                    On_off = folded['On Off Ratio'].iloc[Vr_idx]
                    Vr = folded['Voltage (V)'].iloc[Vr_idx]
                    print("Cycle", cycle_idx, "Threshold Voltage : ", Vth, "reading Voltage :", Vr)
                    
                    # Determine where compliance happens and corresponding voltage
                    below_threshold_idx = folded[folded['On Off Ratio'] < 1.01].index

                    if len(below_threshold_idx) > 0:
                        # Get the index for the 3rd datapoint after the first drop
                        third_point_idx = below_threshold_idx[0] + 2
                        
                        # Check if the third_point_idx is within the DataFrame's bounds
                        if third_point_idx < len(folded):
                            Vcomp = folded.loc[third_point_idx, 'Voltage (V)']
                            
                        else:
                            Vcomp = folded.loc[below_threshold_idx[0]]
        
                    
                    # Append results to arrays
                    Vset_cycle.append(Vth)
                    on_off_cycle.append(On_off)
                    Vread_cycle.append(Vr)
                    Vcomp_cycle.append(Vcomp)                   
                                
            return [Vset_cycle, on_off_cycle, Vread_cycle, Vcomp_cycle]
        
        except Exception as e :
    
            print(f"BADDATA encountered in {csvpath}, skipping the file", e)
            
            Vset_cycle = None
            on_off_cycle = None
            Vread_cycle = None 
            Vcomp_cycle = None
            
            return [Vset_cycle, on_off_cycle, Vread_cycle, Vcomp_cycle]
            
    def quick_pulse_analysis(self, pulse_data):
                
        '''
        Calculates the on-off ratio for a memristor device using pandas DataFrame.
    
        Args:
            pulse_data: A list of lists, where each inner list represents a data point 
                        and contains [time, voltage, current].
    
        Returns:
            A tuple containing:
                - Average on-off ratio.
                - Standard deviation of on-off ratios.
        '''
        df = pd.DataFrame(pulse_data, columns=['Time(T)', 'Voltage (V)', 'Current (A)'])
    
        # Find indices of read voltages
        read_indices = df[df['Voltage (V)'] == df['Voltage (V)'].iloc[1]].index
    
        # Initialize list to store on-off ratios
        on_off_ratios = []
    
        # Calculate on-off ratios for each cycle
        for i in range(0, len(read_indices), 8):
            read1_index = read_indices[i]
            read2_index = read_indices[i + 4]
            on_curr = float(df['Current (A)'].iloc[read1_index])
            off_curr = float(df['Current (A)'].iloc[read2_index])
            print("on :", on_curr , "off:", off_curr )
            on_off_ratio = on_curr/off_curr
            on_off_ratios.append(on_off_ratio)
    
        # Calculate average and standard deviation
        average_ratio = np.mean(on_off_ratios)
        std_dev = np.std(on_off_ratios)
    
        return average_ratio, std_dev
       
    def process_device_data(self, folder_path, grid_file_path):
        """
        Processes IV sweep data for devices in a given folder.

        Args:
            folder_path: Path to the folder containing CSV files for each device.
            grid_file_path: Path to the grid file containing device information.

        Returns:
            A DataFrame with updated grid information, including health scores.
        """

        grid_df = pd.read_csv(grid_file_path)

        # Initialize lists to store relevant data
        health_scores = []

        for index, row in grid_df.iterrows():
            device_number = row['Device']  # Assuming 'Device Number' is the column name
            csv_file_path = os.path.join(folder_path, f"device_{device_number}.csv")  # Adjust filename format as needed

            df = pd.read_csv(csv_file_path)
            contact_current = df['Contact Current (A)'].iloc[0]
            z_position = df['Z position'].iloc[0]

            try:
                # Call IV_calculations function
                Vset_cycle, on_off_cycle, Vread_cycle, Vcomp_cycle = self.IV_calculations(csv_file_path) 

                if Vset_cycle is not None:
                    # Calculate averages and standard deviations
                    avg_Vset = np.mean(Vset_cycle)
                    std_Vset = np.std(Vset_cycle)
                    avg_on_off = np.mean(on_off_cycle)
                    std_on_off = np.std(on_off_cycle)
                    avg_Vread = np.mean(Vread_cycle)
                    std_Vread = np.std(Vread_cycle)

                    # Update grid DataFrame 
                    grid_df.loc[index, 'Vset_Avg'] = avg_Vset
                    grid_df.loc[index, 'Vset_Std'] = std_Vset
                    grid_df.loc[index, 'On_Off_Avg'] = avg_on_off
                    grid_df.loc[index, 'On_Off_Std'] = std_on_off
                    grid_df.loc[index, 'Vread_Avg'] = avg_Vread
                    grid_df.loc[index, 'Vread_Std'] = std_Vread
                    grid_df.loc[index, 'Contact Current'] = contact_current
                    grid_df.loc[index, 'Z Position'] = z_position
                    
                    if any(on_off < 10 for on_off in on_off_cycle):
                        health_scores.append(0) 
                    # elif any(Vset_cycle[i] > Vcomp_cycle[i] for i in range(len(Vset_cycle))):
                    #     health_scores.append(0)
                    # elif any(Vread_cycle[i] > Vset_cycle[i] for i in range(len(Vset_cycle))):
                    #     health_scores.append(0)
                    else :
                        health_scores.append(100)
                        

            except Exception as e:
                print(f"Error processing device {device_number}: {e}")


        normalized_avg_on_off = ((grid_df['On_Off_Avg'] - grid_df['On_Off_Avg'].min()) / (grid_df['On_Off_Avg'].max() - grid_df['On_Off_Avg'].min())).tolist()
        normalized_std_on_off = ((grid_df['On_Off_Std'] - grid_df['On_Off_Std'].min()) / (grid_df['On_Off_Std'].max() - grid_df['On_Off_Std'].min())).tolist()
        normalized_std_Vset = ((grid_df['Vset_Avg'] - grid_df['Vset_Avg'].min()) / (grid_df['Vset_Avg'].max() - grid_df['Vset_Std'].min())).tolist()

        # Calculate health score 
        for i, score in enumerate(health_scores):
            if score != 0:
                health_scores[i] = 2*normalized_avg_on_off[i] - 1*normalized_std_on_off[i] - 0.5*normalized_std_Vset[i]

            else:
                health_scores[i] = None

        # Add normalized health scores to the grid DataFrame
        grid_df['Health Score'] = health_scores
        print("Finished processing all the devices in the grid")
        return grid_df


class Listmaker():
    
    def generate_voltage_data(self,forward_voltage, reset_voltage, step_voltage, timer_delay, forming_cycle, forming_voltage, cycles):
        voltages = []
        times = []
        current_time = 0
    
        # Forming cycle if needed
        if forming_cycle:
            voltage = 0
            while voltage <= forming_voltage:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage += step_voltage
            
            voltage -= step_voltage
            while voltage >= 0:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage -= step_voltage
            
            # Immediate reset sweep after forming cycle
            voltage = 0
            while voltage >= reset_voltage:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage -= step_voltage
    
            voltage += step_voltage
            while voltage <= 0:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage += step_voltage
    
        # Regular cycles
        for _ in range(cycles):
            # Forward voltage cycle
            voltage = 0
            while voltage <= forward_voltage:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage += step_voltage
    
            voltage -= step_voltage
            while voltage >= 0:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage -= step_voltage
    
            # Reset voltage cycle
            voltage = 0
            while voltage >= reset_voltage:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage -= step_voltage
    
            voltage += step_voltage
            while voltage <= 0:
                voltages.append(voltage)
                times.append(current_time)
                current_time += timer_delay
                voltage += step_voltage
    
        return times, voltages

    def generate_pulsing_data(self, write_pulses, write_voltage, write_width, read_pulses, read_voltage, read_width, erase_pulses, erase_voltage, erase_width, cycles):
        voltages = []
        times = []
        current_time = 0
    
        for _ in range(cycles):
            # Write pulses
            for _ in range(write_pulses):
                voltages.append(write_voltage)
                times.append(current_time)
                current_time += write_width
                voltages.append(0)
                times.append(current_time)
                current_time += write_width
            
            # Read pulses
            for _ in range(read_pulses):
                voltages.append(read_voltage)
                times.append(current_time)
                current_time += read_width
                voltages.append(0)
                times.append(current_time)
                current_time += read_width
            
            # Erase pulses
            for _ in range(erase_pulses):
                voltages.append(erase_voltage)
                times.append(current_time)
                current_time += erase_width
                voltages.append(0)
                times.append(current_time)
                current_time += erase_width
            
            # Read pulses again
            for _ in range(read_pulses):
                voltages.append(read_voltage)
                times.append(current_time)
                current_time += read_width
                voltages.append(0)
                times.append(current_time)
                current_time += read_width
    
        return times, voltages

    def save_sweep_to_csv(self, times, voltages, filename):
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Time (s)', 'Voltage (V)'])
            for t, v in zip(times, voltages):
                csvwriter.writerow([t, v])
       
    def make_updated_sweep(self, base_file_path, fwd_volt, reset_volt, step_voltage, timer_delay):

        """
        Generates an updated voltage sweep CSV file based on a base sweep file. 
        Retains the forming cycle from the base file if detected, and appends new cycles with specified parameters.
    
        Args:
            base_file_path (str): Path to the base sweep CSV file.
            forward_voltage (float): Maximum forward voltage for the new cycles.
            reset_voltage (float): Minimum reset voltage for the new cycles.
            step_voltage (float): Voltage step size for each cycle.
            timer_delay (float): Time delay for each voltage step.

        """
        # Read the base file
        base_df = pd.read_csv(base_file_path)
        base_dir, base_name = os.path.split(base_file_path)
        name, ext = os.path.splitext(base_name)
        output_file = os.path.join(base_dir, f"{name}_adjusted{ext}")

    
        # Check for forming cycle
        voltages = base_df['Voltage (V)']
        max_voltages = []
        start_index = 0
        forming_cycle = False
        forming_voltage = None
    
        # Detect cycles and identify forming voltage
        cycles = -1
        for i in range(1, len(voltages)):
            if voltages.iloc[i - 1] == 0 and voltages.iloc[i] > 0:
                # Cycle detected
                cycles += 1
                if start_index != i - 1:
                    cycle_data = voltages.iloc[start_index:i].reset_index(drop=True)
                    max_voltages.append(cycle_data.max())
                start_index = i
    
        # Add the last cycle if not already added
        if start_index < len(voltages):
            cycles += 1
            cycle_data = voltages.iloc[start_index:].reset_index(drop=True)
            max_voltages.append(cycle_data.max())
    
        # Check for forming cycle
        if max_voltages and max_voltages[0] > max(max_voltages[1:], default=-float('inf')):
            forming_cycle = True
            forming_voltage = max_voltages[0]
    
        print(f"Number of cycles in the base file: {cycles}")
    
        # Generate new sweep data
        times, new_voltages = [], []
        
        if forming_cycle:
            print(f"Forming cycle detected with forming voltage: {forming_voltage}")
            times, new_voltages = self.generate_voltage_data(
                forward_voltage = fwd_volt,
                reset_voltage = reset_volt,
                step_voltage = step_voltage,
                timer_delay = timer_delay,
                forming_cycle=True,
                forming_voltage=forming_voltage,
                cycles=cycles-1
            )
            #print(times,voltages)
            
            # Save updated sweep to CSV
            self.save_sweep_to_csv(times, new_voltages, output_file)
    
        else :
            times, voltages = self.generate_voltage_data(
                forward_voltage=fwd_volt,
                reset_voltage=reset_volt,
                step_voltage=step_voltage,
                timer_delay=timer_delay,
                forming_cycle=False,
                forming_voltage=0,
                cycles=cycles
            )
            
            # Save updated sweep to CSV
            self.save_sweep_to_csv(times, new_voltages, output_file)

        print(f"Updated sweep saved to {output_file}")
        
        

        return output_file

    