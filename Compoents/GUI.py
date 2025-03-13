import PySimpleGUI as sg
import cv2
import threading

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
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window.write_event_value('-CAMERA-', imgbytes)

# Function to stop the camera feed
def stop_camera():
    global camera, camera_running
    camera_running = False
    if camera:
        camera.release()
        camera = None

# Movement tab layout
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

# Grid tab layout
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

# Settings tab layout
settings_tab = [
    [sg.Text("Sweep Settings")],
    [sg.Text("Forward Voltage:"), sg.InputText(key='-FORWARD_VOLTAGE-', size=(10, 1))],
    [sg.Text("Reset Voltage:"), sg.InputText(key='-RESET_VOLTAGE-', size=(10, 1))],
    [sg.Text("Step Voltage:"), sg.InputText(key='-STEP_VOLTAGE-', size=(10, 1))],
    [sg.Text("Timer Delay:"), sg.InputText(key='-TIMER_DELAY-', size=(10, 1))],
    [sg.Text("Forming Cycle Needed (y/n):"), sg.InputText(key='-FORMING_CYCLE-', size=(10, 1))],
    [sg.Text("Forming Voltage:"), sg.InputText(key='-FORMING_VOLTAGE-', size=(10, 1))],
    [sg.Text("Number of Cycles:"), sg.InputText(key='-CYCLES-', size=(10, 1))],
    [sg.Button("Save Settings", key='-SAVE_SETTINGS-')],
]

# Main layout with tabs
layout = [
    [sg.TabGroup([
        [sg.Tab('Movement', movement_tab), 
         sg.Tab('Grid', grid_tab), 
         sg.Tab('Settings', settings_tab)]
    ])],
]

# Create the window
window = sg.Window("Stage Controller with Camera and Lights", layout, resizable=True)

while True:
    event, values = window.read(timeout=10)
    
    if event == sg.WINDOW_CLOSED:
        stop_camera()  # Ensure camera is stopped before exiting
        break
    
    if event == '-START_CAMERA-':
        if camera_thread is None or not camera_thread.is_alive():
            camera_thread = threading.Thread(target=start_camera, daemon=True)
            camera_thread.start()

    if event == '-STOP_CAMERA-':
        stop_camera()
        
    if event == '-CAMERA-':
        imgbytes = values[event]
        window['-CAMERA_FEED-'].update(data=imgbytes)
        window['-GRID_CAMERA_FEED-'].update(data=imgbytes)
    
    # Handle other events...
    # ...existing code...
