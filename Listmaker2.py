# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:06:03 2024

@author: shrey
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def generate_voltage_data(forward_voltage, reset_voltage, step_voltage, timer_delay, forming_cycle, forming_voltage, cycles):
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

def save_to_csv(times, voltages, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Time (s)', 'Voltage (V)'])
        for t, v in zip(times, voltages):
            csvwriter.writerow([t, v])

def plot_data(times, voltages):
    plt.figure()
    plt.plot(times, voltages, marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Time vs Voltage')
    plt.grid(True)
    return plt.gcf()

def draw_figure(canvas, figure):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def generate_pulsing_data(write_pulses, write_voltage, write_width, read_pulses, read_voltage, read_width, erase_pulses, erase_voltage, erase_width, cycles):
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

# GUI layout
iv_sweep_layout = [
    [sg.Text('Enter the forward voltage:'), sg.InputText(key='forward_voltage')],
    [sg.Text('Enter the reset voltage (negative value):'), sg.InputText(key='reset_voltage')],
    [sg.Text('Enter the step voltage:'), sg.InputText(key='step_voltage')],
    [sg.Text('Enter the timer delay in seconds:'), sg.InputText(key='timer_delay')],
    [sg.Text('Is forming cycle needed (y/n)?'), sg.InputText(key='forming_cycle')],
    [sg.Text('Enter the forming voltage (if applicable):'), sg.InputText(key='forming_voltage')],
    [sg.Text('Enter the number of cycles (default 1 or 2):'), sg.InputText(key='cycles')],
    [sg.Button('Visualize'), sg.Button('Save Sweep')],
    [sg.Canvas(key='canvas')],
]

pulsing_layout = [
    [sg.Text('Write Pulse Settings')],
    [sg.Text('Number of pulses:'), sg.InputText(key='write_pulses', size=(20,10)),
     sg.Text('Voltage (V):'), sg.InputText(key='write_voltage', size=(20,10)),
     sg.Text('Pulse width (s):'), sg.InputText(key='write_width', size=(20,10))],
    [sg.Text('Read Pulse Settings')],
    [sg.Text('Number of pulses:'), sg.InputText(key='read_pulses',size=(20,10)),
     sg.Text('Voltage (V):'), sg.InputText(key='read_voltage',size=(20,10)),
     sg.Text('Pulse width (s):'), sg.InputText(key='read_width',size=(20,10))],
    [sg.Text('Erase Pulse Settings')],
    [sg.Text('Number of pulses:'), sg.InputText(key='erase_pulses',size=(20,10)),
     sg.Text('Voltage (V):'), sg.InputText(key='erase_voltage',size=(20,10)),
     sg.Text('Pulse width (s):'), sg.InputText(key='erase_width',size=(20,10))],
    [sg.Text('Number of cycles:'), sg.InputText(key='pulse_cycles')],
    [sg.Button('Generate Pulses'), sg.Button('Save Pulses')],
    [sg.Canvas(key='pulsing_canvas')],
]

layout = [
    [sg.TabGroup([
        [sg.Tab('I-V Sweep', iv_sweep_layout), sg.Tab('Pulsing', pulsing_layout)]
    ])]
]

# Create the Window
window = sg.Window('Voltage Sweep Generator', layout, finalize=True)

times = []
voltages = []

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == 'Visualize':
        forward_voltage = float(values['forward_voltage'])
        reset_voltage = float(values['reset_voltage'])
        step_voltage = float(values['step_voltage'])
        timer_delay = float(values['timer_delay'])
        forming_cycle = values['forming_cycle'].lower() == 'y'
        forming_voltage = float(values['forming_voltage']) if forming_cycle else None
        cycles = int(values['cycles']) if values['cycles'] else (2 if forming_cycle else 1)
        
        times, voltages = generate_voltage_data(forward_voltage, reset_voltage, step_voltage, timer_delay, forming_cycle, forming_voltage, cycles)
        figure = plot_data(times, voltages)
        draw_figure(window['canvas'].TKCanvas, figure)
    if event == 'Save Sweep':
        if times and voltages:
            save_path = sg.popup_get_file('Save as', save_as=True, no_window=True, file_types=(("CSV Files", "*.csv"),))
            if save_path:
                save_to_csv(times, voltages, save_path)
        else:
            sg.popup('No data to save. Please generate the data first by clicking "Visualize".')
    if event == 'Generate Pulses':
        write_pulses = int(values['write_pulses'])
        write_voltage = float(values['write_voltage'])
        write_width = float(values['write_width'])
        read_pulses = int(values['read_pulses'])
        read_voltage = float(values['read_voltage'])
        read_width = float(values['read_width'])
        erase_pulses = int(values['erase_pulses'])
        erase_voltage = float(values['erase_voltage'])
        erase_width = float(values['erase_width'])
        pulse_cycles = int(values['pulse_cycles'])
        
        times, voltages = generate_pulsing_data(write_pulses, write_voltage, write_width, read_pulses, read_voltage, read_width, erase_pulses, erase_voltage, erase_width, pulse_cycles)
        figure = plot_data(times, voltages)
        draw_figure(window['pulsing_canvas'].TKCanvas, figure)
    if event == 'Save Pulses':
        if times and voltages:
            save_path = sg.popup_get_file('Save as', save_as=True, no_window=True, file_types=(("CSV Files", "*.csv"),))
            if save_path:
                save_to_csv(times, voltages, save_path)
        else:
            sg.popup('No data to save. Please generate the data first by clicking "Generate Pulses".')

window.close()
