# -*- coding: utf-8 -*-
"""
Modified on Thu Dec 7, 2024

@author: amdm
"""
import os
import PySimpleGUI as sg
from time import sleep
import cv2
import numpy as np
from zaber_motion import Library, Units
from zaber_motion.ascii import Connection, Axis
from Openflexture import stage
from Openflexture import Light  # Import the Light class
import threading
from Vision import overlay
import csv
import math
from Source_Measure_Unit import KeysightSMU, pyvisa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm



# Initialize global objects for both Arduino and Zaber stages
stage_obj = None
zaber_x_axis = None
zaber_y_axis = None
camera = None
camera_thread = None
camera_running = False
light_controller = None  # Initialize Light with default port
SMU_adress = None
#measurement = measurement()

global save_directory
save_directory = "C:/Users/amdm/OneDrive - Nanyang Technological University/ViPSA data folder/Spinbot_memristors_PEA/opt1"
sweep_path = "C:/Users/amdm/Desktop/sweep patterns/Sweep_2Dmems_faster.csv"
pulse_path = "C:/Users/amdm/Desktop/sweep patterns/Pulse20.csv"

# Function to connect to Arduino stage
def connect_to_stage(values):
    
    global stage_obj
    port = values['-PORT-']
    baudrate = int(values['-BAUD-'])
    multiplier = float(values['-SCALE-'])
    stage_obj = stage(port, baudrate, multiplier)

# Function to handle Arduino stage movement
def handle_move_arduino(direction, steps):
    if stage_obj:
        if direction == 'X':
            stage_obj.move_x_by(steps)
        elif direction == 'Y':
            stage_obj.move_y_by(steps)
        elif direction == 'Z':
            stage_obj.move_z_by(steps)

# Function to connect to Zaber stage
def connect_to_zaber(port):
    global zaber_x_axis, zaber_y_axis
    connection = Connection.open_serial_port(port)
    device_list = connection.detect_devices()
    print("Found {} devices".format(len(device_list)))
    print(device_list)

    # Assigning the separate axes to separate devices
    zaber_x_axis = device_list[1].get_axis(1)
    zaber_y_axis = device_list[0].get_axis(1)

# Function to handle Zaber stage movement
def handle_move_zaber(axis, distance):
    try:
        if axis == 'X' and zaber_x_axis:
            zaber_x_axis.move_relative(distance)
        elif axis == 'Y' and zaber_y_axis:
            zaber_y_axis.move_relative(distance)
    except Exception as e:
        print(e)

# Function to start the camera feed
def start_camera():
    global camera, camera_running
    camera = cv2.VideoCapture(0)
    camera_running = True
    while camera_running:
        ret, frame = camera.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        framex = overlay(frame)
        imgbytes = cv2.imencode('.png', framex)[1].tobytes()
        window.write_event_value('-CAMERA-', imgbytes)

# Function to stop the camera feed
def stop_camera():
    global camera, camera_running
    camera_running = False
    if camera:
        camera.release()
        camera = None

# Function to generate grid and save to CSV
def make_grid(x_distance, y_distance, rows, cols, save_path, tilt_angle=0):
    """
    Generate a grid of device positions with an optional tilt angle, including Z-height interpolation.
    
    Args:
        x_distance (float): Distance between device centers along the X-axis.
        y_distance (float): Distance between device centers along the Y-axis.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        save_path (str): Path to save the grid CSV file.
        tilt_angle (float): Tilt angle in degrees (default: 0°).
    """
    # Get the starting position from Zaber stages
    start_x = zaber_x_axis.get_position()
    start_y = zaber_y_axis.get_position()

    grid = []
    device_number = 1
    for row in range(rows):
        for col in range(cols):
            # Compute the original unrotated coordinates
            x_coord = start_x + col * x_distance
            y_coord = start_y + row * y_distance
            
            # Add the device to the grid
            grid.append([device_number, x_coord, y_coord])
            device_number += 1

    # Save the grid to a CSV file
    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Device', 'X', 'Y'])
        writer.writerows(grid)
    
    print(f"Grid with tilt angle {tilt_angle}° saved to {save_path}")
    
def save_file(data, data_name, sample_no, device_no, cont_current, Z_pos):
    directory_path = f"{save_directory}/slot_{sample_no}/{data_name}"
    os.makedirs(directory_path, exist_ok=True)
    file_path = os.path.join(directory_path, f"device_{device_no}.csv")
    df = pd.DataFrame(data, columns=['Time(T)', 'Voltage (V)', 'Current (A)'])
    df['Contact Current (A)'] = cont_current
    df['Z movement'] = Z_pos
    df.to_csv(file_path, index=True)
    print(f"Data saved to {file_path}")
    
    return file_path

def show_plot_windowed(csvpath):
    
    def plot_cycles(csvpath):
        # Read the CSV file
        df = pd.read_csv(csvpath)
        
        # Convert negative current values to their absolute values
        df['Current (A)'] = df['Current (A)'].abs()
        
        # Initialize a figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Initialize variables to keep track of the start of each cycle
        colormap = cm.get_cmap('Set1')  # Choose a colormap
        num_colors = 6  # Adjust to match the number of cycles you want to plot
        
        # Generate a list of colors from the colormap
        colors = [colormap(i / num_colors) for i in range(num_colors)]
        
        start_index = 0
        color_idx = 0  # Initialize color index
        legend_labels = []  # List to store legend labels
        
        # Iterate through the DataFrame to find cycles
        for i in range(1, len(df)):
            if df['Voltage (V)'].iloc[i-1] == 0 and df['Voltage (V)'].iloc[i] > 0:
                if start_index != i-1:
                    ax.plot(df['Voltage (V)'].iloc[start_index:i], 
                            df['Current (A)'].iloc[start_index:i], 
                            linestyle='-', 
                            color=colors[color_idx % len(colors)], 
                            label=f'Cycle {color_idx + 1}')
                    legend_labels.append(f'Cycle {color_idx + 1}')
                    color_idx += 1
                start_index = i
        
        # Plot the remaining cycle, if any
        if start_index < len(df):
            ax.plot(df['Voltage (V)'].iloc[start_index:], 
                    df['Current (A)'].iloc[start_index:], 
                    linestyle='-', 
                    color=colors[color_idx % len(colors)], 
                    label=f'Cycle {color_idx + 1}')
            legend_labels.append(f'Cycle {color_idx + 1}')
        
        Curr = df["Contact Current (A)"].iloc[1]
        
        # Set axis labels and logarithmic scale
        ax.set_ylabel('Current (log scale)')
        ax.set_xlabel('Voltage')
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend(title="Cycles")
        ax.set_title(f"contact_res = {Curr}")
        
        return fig
    
    def draw_figure(canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
        return figure_canvas_agg
    
    # PySimpleGUI layout
    layout = [
        [sg.Text(f"Displaying plot for {csvpath}")],
        [sg.Canvas(key="-CANVAS-")],
        [sg.Button("Close")]
    ]
    
    # Create the PySimpleGUI Window
    window = sg.Window("Cycle Plotter", layout, finalize=True)
    
    # Generate and display the plot
    fig = plot_cycles(csvpath)
    draw_figure(window["-CANVAS-"].Widget, fig)
    
    while True:
        event, _ = window.read()
        if event in (sg.WINDOW_CLOSED, "Close"):
            break
    
    window.close()
 
def single_measurement(sample_no, device_no):
    SMU = KeysightSMU(0)
    adr = SMU.get_address()
    rm = pyvisa.ResourceManager()
    SMU_1 = rm.open_resource(adr)
    
    cont_current = abs(float(SMU_1.query('MEAS:CURR?')))
    dist = stage_obj.get_current_position()[2]
    
    sweep_data = SMU.list_IV_sweep_manual(sweep_path, 0.0001, 0.01, delay=0.0, adr=SMU_adress)
    saved_file = save_file(sweep_data,"Sweep", sample_no, device_no, cont_current, dist)
        
    show_plot_windowed(saved_file)
    #show_plot(saved_file_p)
            

'''
# Function to handle the measurement loop
def handle_measurement_loop(gridpath, startpoint, skip_instances):
    try:
        grid = np.genfromtxt(gridpath, delimiter=',', skip_header=1)
        device_number, x_coord, y_coord = grid[startpoint]
        zaber_x_axis.move_absolute(x_coord)
        zaber_y_axis.move_absolute(y_coord)

        # Replace with the desired slot name or pass it as an input
        measurement.Measure_gridwise("slot7", grid, startpoint, skip_instances=skip_instances)
    except Exception as e:
        print(f"Error in measurement loop: {e}")
'''


# Main function
def main():
    global window, camera_thread, light_controller

    # Tab layouts
    movement_tab = [
        [sg.Column([
            [sg.Text("Camera Feed")],
            [sg.Image(filename='', key='-CAMERA_FEED-')],
            [sg.Button("Start Camera", key='-START_CAMERA-'), sg.Button("Stop Camera", key='-STOP_CAMERA-')],
        ], vertical_alignment='top'),
        sg.Column([
            [sg.Text("Arduino Stage Control")],
            [sg.Text("Port:"), sg.InputText(default_text='COM4', key='-PORT-', size=(10, 1))],
            [sg.Text("Baudrate:"), sg.InputText(default_text='115200', key='-BAUD-', size=(10, 1))],
            [sg.Text("Scale (multiplier):"), sg.InputText(default_text='1', key='-SCALE-', size=(10, 1))],
            [sg.Button("Connect Arduino", key='-CONNECT-')],
            
            [sg.HorizontalSeparator()],
            
            [sg.Text("Move Arduino X Axis by Steps:"), sg.InputText(default_text='0', key='-X_STEPS-', size=(10, 1)), sg.Button("Move X+", key='-MOVE_X_POS-'), sg.Button("Move X-", key='-MOVE_X_NEG-')],
            [sg.Text("Move Arduino Y Axis by Steps:"), sg.InputText(default_text='0', key='-Y_STEPS-', size=(10, 1)), sg.Button("Move Y+", key='-MOVE_Y_POS-'), sg.Button("Move Y-", key='-MOVE_Y_NEG-')],
            [sg.Text("Move Arduino Z Axis by Steps:"), sg.InputText(default_text='0', key='-Z_STEPS-', size=(10, 1)), sg.Button("Move Z+", key='-MOVE_Z_POS-'), sg.Button("Move Z-", key='-MOVE_Z_NEG-')],
            
            [sg.HorizontalSeparator()],
            
            [sg.Button("Set Zero", key='-SET_ZERO-'), sg.Button("Go to Zero", key='-GO_ZERO-')],
            [sg.Button("Disconnect Arduino", key='-DISCONNECT-')],
            
            [sg.HorizontalSeparator()],
            
            [sg.Text("Zaber Stage Control")],
            [sg.Text("Zaber Port:"), sg.InputText(default_text='COM6', key='-ZABER_PORT-', size=(10, 1))],
            [sg.Button("Connect Zaber", key='-CONNECT_ZABER-')],
            
            [sg.Text("Move Zaber X Axis by Distance:"), sg.InputText(default_text='0', key='-ZABER_X_DIST-', size=(10, 1)), sg.Button("Move X+", key='-MOVE_ZABER_X_POS-'), sg.Button("Move X-", key='-MOVE_ZABER_X_NEG-')],
            [sg.Text("Move Zaber Y Axis by Distance:"), sg.InputText(default_text='0', key='-ZABER_Y_DIST-', size=(10, 1)), sg.Button("Move Y+", key='-MOVE_ZABER_Y_POS-'), sg.Button("Move Y-", key='-MOVE_ZABER_Y_NEG-')],
            
            [sg.Button("Disconnect Zaber", key='-DISCONNECT_ZABER-')],
            
            [sg.HorizontalSeparator()],
            
            [sg.Text("Lights Control")],
            [sg.Button("Connect Lights", key='-CONNECT_LIGHTS-')],
            [sg.Button("Turn On Lights", key='-TURN_ON_LIGHTS-')],
            [sg.Button("Turn Off Lights", key='-TURN_OFF_LIGHTS-')],
        ], vertical_alignment='top')],
    ]

    grid_tab = [
        [sg.Column([
            [sg.Text("Camera Feed")],
            [sg.Image(filename='', key='-CAMERA_FEED-')],
            [sg.Button("Start Camera", key='-START_CAMERA-'), sg.Button("Stop Camera", key='-STOP_CAMERA-')],
        ], vertical_alignment='top'),
        sg.Column([
            [sg.Text("Grid Creation")],
            [sg.Text("X Distance:"), sg.InputText(key='-X_DIST-', size=(10, 1))],
            [sg.Text("Y Distance:"), sg.InputText(key='-Y_DIST-', size=(10, 1))],
            [sg.Text("Rows:"), sg.InputText(key='-ROWS-', size=(10, 1))],
            [sg.Text("Columns:"), sg.InputText(key='-COLS-', size=(10, 1))],
            [sg.Text("Tilt:"), sg.InputText(key='-TILT-', size=(10, 1)), sg.Text("degrees")],
            [sg.Text("Save Path:"), sg.InputText(key='-SAVE_PATH-', size=(40, 1)), sg.FileSaveAs()],
            [sg.Button("Create Grid", key='-CREATE_GRID-')],
            [sg.Image(filename='', key='-GRID_CAMERA_FEED-')]
        ], vertical_alignment='top')], 
    ]

    measure_tab = [
        
        [sg.Text("Single Measurement")],
        [sg.Text("Sample # :"),sg.InputText(default_text='0', key='-SAMPLE_ID-', size=(10, 1)),
         sg.Text("Device # :"),sg.InputText(default_text='0', key='-DEVICE_ID-', size=(10, 1))],
        [sg.Text("Contact ?"), sg.Radio("Yes",  "contact", key='-CONTACTED-'), sg.Radio("No", "contact", key='-NOT_CONTACTED-')],
        [sg.Button("Run Single Measurement", key='-RUN_SINGLE_MEASUREMENT-')],
        [sg.Button("Approach", key='-APPROACH-')],
        
        [sg.HorizontalSeparator()],
        
        #[sg.Text("Measurement Loop")],
        #[sg.Button("Connect to measure", key='-CONNECT_ALL-')],
        #[sg.Text("Grid Path"), sg.InputText(key='-GRID_PATH-', size=(40, 1)), sg.FileBrowse()],
        #[sg.Text("Start Point"), sg.InputText(default_text='0', key='-STARTPOINT-', size=(10, 1))],
        #[sg.Text("Skip Instances"), sg.InputText(default_text='0', key='-SKIP_INSTANCES-', size=(10, 1))],
        #[sg.Button("Run Grid Measurement", key='-RUN_GRID_MEASUREMENT-')],

    ]
    

    # Main layout with tabs
    
    layout = [
        [sg.TabGroup([
            [sg.Tab('Movement', movement_tab), 
             sg.Tab('Grid', grid_tab), 
             sg.Tab('Measure', measure_tab)]
        ])],
    ]

    # Create the window
    window = sg.Window("Stage Controller with Camera and Lights", layout, resizable=True)

    while True:
        event, values = window.read(timeout=10)
        
        if event == sg.WINDOW_CLOSED:
            if stage_obj:
                stage_obj.disconnect()
            if zaber_x_axis:
                zaber_x_axis.device.connection.close()
            if zaber_y_axis:
                zaber_y_axis.device.connection.close()
                print("Zaber devices disconnected")
            
            if light_controller :
                light_controller.disconnect()
            
            stop_camera()  # Ensure camera is stopped before exiting
            break
        
        if event == '-CONNECT-':
            connect_to_stage(values)
        
        if event == '-MOVE_X_POS-':
            handle_move_arduino('X', float(values['-X_STEPS-']))
        if event == '-MOVE_X_NEG-':
            handle_move_arduino('X', -float(values['-X_STEPS-']))

        if event == '-MOVE_Y_POS-':
            handle_move_arduino('Y', float(values['-Y_STEPS-']))
        if event == '-MOVE_Y_NEG-':
            handle_move_arduino('Y', -float(values['-Y_STEPS-']))

        if event == '-MOVE_Z_POS-':
            handle_move_arduino('Z', float(values['-Z_STEPS-']))
        if event == '-MOVE_Z_NEG-':
            handle_move_arduino('Z', -float(values['-Z_STEPS-']))

        if event == '-SET_ZERO-':
            if stage_obj:
                stage_obj.set_zero()
        
        if event == '-GO_ZERO-':
            if stage_obj:
                stage_obj.go_to_zero()
                
        if event == '-DISCONNECT-':
            if stage_obj:
                stage_obj.disconnect()
                
        # Handle Zaber stage events
        if event == '-CONNECT_ZABER-':
            connect_to_zaber(values['-ZABER_PORT-'])
        
        if event == '-MOVE_ZABER_X_POS-':
            handle_move_zaber('X', float(values['-ZABER_X_DIST-']))
        if event == '-MOVE_ZABER_X_NEG-':
            handle_move_zaber('X', -float(values['-ZABER_X_DIST-']))

        if event == '-MOVE_ZABER_Y_POS-':
            handle_move_zaber('Y', float(values['-ZABER_Y_DIST-']))
        if event == '-MOVE_ZABER_Y_NEG-':
            handle_move_zaber('Y', -float(values['-ZABER_Y_DIST-']))

        if event == '-DISCONNECT_ZABER-':
            if zaber_x_axis:
                zaber_x_axis.device.connection.close()
            if zaber_y_axis:
                zaber_y_axis.device.connection.close()
            print("Zaber devices disconnected")
        
        # Handle camera events
        if event == '-START_CAMERA-':
            if camera_thread is None or not camera_thread.is_alive():
                camera_thread = threading.Thread(target=start_camera, daemon=True)
                camera_thread.start()

        if event == '-STOP_CAMERA-':
            stop_camera()
            
        
        if event == '-CAMERA-':
            imgbytes = values[event]
            window['-CAMERA_FEED-'].update(data=imgbytes)
        
        # Handle light events
        if event == '-CONNECT_LIGHTS-':
            light_controller = Light()
            
        if event == '-TURN_ON_LIGHTS-':
            light_controller.control_lights('on')

        if event == '-TURN_OFF_LIGHTS-':
            
            light_controller.control_lights('off')

        # Handle grid creation
        if event == '-CREATE_GRID-':
            x_dist = float(values['-X_DIST-'])
            y_dist = float(values['-Y_DIST-'])
            
            rows = int(values['-ROWS-'])
            cols = int(values['-COLS-'])
            tilt = float(values['-TILT-'])
            save_path = values['-SAVE_PATH-']
            make_grid(x_dist, y_dist, rows, cols, save_path, tilt)


        # Handle camera events
        if event == '-START_CAMERA-':
            if camera_thread is None or not camera_thread.is_alive():
                camera_thread = threading.Thread(target=start_camera, daemon=True)
                camera_thread.start()
                
                
        if event == '-STOP_CAMERA-':
            stop_camera()

        if event == '-CAMERA-' or event == '-GRID_CAMERA_FEED-':
            imgbytes = values[event]
            window['-CAMERA_FEED-'].update(data=imgbytes)
            window['-GRID_CAMERA_FEED-'].update(data=imgbytes)
           
        if event == '-RUN_SINGLE_MEASUREMENT-':
            if values['-CONTACTED-'] :
                sample_no = values['-SAMPLE_ID-']
                device_no = values['-DEVICE_ID-']
                single_measurement(sample_no, device_no)
                
            elif values['-NOT_CONTACTED-']:
                sg.popup("Please contact with the device first !")
            
            
        if event == '-CONNECT_SMU-':
            SMU_adress = KeysightSMU() 
       
        '''    
        if event == '-RUN_GRID_MEASUREMENT-':
            gridpath = values['-GRID_PATH-']
            startpoint = int(values['-STARTPOINT-'])
            skip_instances = int(values['-SKIP_INSTANCES-'])
            handle_measurement_loop(gridpath, startpoint, skip_instances)
            

             
        if event == '-CONNECT_ALL-' :
            measurement.connect_all()
                        
        if event == '-APPROACH-':
            measurement.detect_contact_and_move_z(SMU_adress, step=1)
        
        if event == '-MEASURE_HERE-':
            sample_no = values['-SAMPLE_NO-']
            device_no = values['-DEVICE_NO-']
            measurement.measure_and_save(sample_no, device_no)
        '''

if __name__ == '__main__':
    main()
