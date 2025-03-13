# -*- coding: utf-8 -*-
"""
Shreyas' code.

This module holds the instrument class for tools from Keysight.
Methods are written in SCPI. 
Make sure your environment is set to Python 3.10 and the correct drivers from keysight website are downloaded and installed.

Classes:
    stage()
        Openflexture stage runs on grbl 
        
    LED()
        there are a few open unused pins on the hat.
        they can be configured to run LEDs, other servos etc using PWM/Gcode's "feed" settings
        
"""

import serial
import time
from zaber_motion import Library, Units
from zaber_motion.ascii import Connection, AllAxes, Axis, Device

#from controllably.Move import Mover

class stage():

    def __init__(self, port, baudrate, multiplier):
        self.scale = multiplier #this changs according to the zoom level of the optics. 
        self.port = port
        self.baudrate = baudrate
    
        self.x_pos = 0  
        self.y_pos = 0  
        self.z_pos = 0
        
        try:
            self.ser = serial.Serial(port, baudrate, timeout=2)
            time.sleep(2)  # Wait for connection to establish
            print("Connected to Arduino")
        except Exception as e:
            print("Error:", e)
    
    def flush(self):
        if self.ser is None or not self.ser.is_open:
            print("Arduino not connected")
            return
        try : 
            self.ser.flush()
            print("flsuhed")
        
        except Exception as e:
            print("Error:", e)
            
            
    def is_busy(self) : 
        if self.ser is None or not self.ser.is_open:
            print("Arduino not connected")
            return
        try :
            # Send status query
            time.sleep(0.3)
            self.ser.write(b'?')
            time.sleep(0.3)
            status = self.ser.readline().decode('utf-8').strip()
            return status
            
        except Exception as e:
            print("Error:", e)
            
    def change_scale(self, new_scale):
        old_scale = self.scale
        self.scale = new_scale
        print(f"Scaling changed from {old_scale} to {self.scale}")
                   
    def set_zero(self):
        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            gcode = "G10 P0 L20 X0 Y0 Z0\n"
            self.ser.write(gcode.encode())
            self.x_pos = 0  
            self.y_pos = 0  
            self.z_pos = 0
            print("Current position set as zero")
            
        except Exception as e:
            print("Error:", e)
    
    def go_to_zero(self):
        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            gcode = "G90\nG1 X0 F450\nG1 Y0 F450\n"
            self.ser.write(gcode.encode())
            print("going to zero")
            self.x_pos = 0  
            self.y_pos = 0  
            self.z_pos = 0
        except Exception as e:
            print("Error:", e)
    
    def get_current_position(self):
        # Check if the serial connection is open
        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return None, None

        try:
            
            x = self.x_pos
            y = self.y_pos
            z = self.z_pos
            
            return x, y, z
        except Exception as e:
            print("Error querying or parsing position:", e)
            return None, None
    
    
    def move_x_by(self, steps):
        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            distance = steps * self.scale  # Change this value according to the zooming
            feedrate = 450  # Feedrate in mm/minute
            gcode = f"G91\nG1 X{distance} F{feedrate}\nG90\n"
            self.ser.write(gcode.encode())
            time.sleep(abs(distance)*0.15)
            self.ser.flush()
            status = self.is_busy()
            while "Run" in status :
                status = self.is_busy()
                print (status, "still running")
            
            self.x_pos = self.x_pos + steps 
            print(f"Moved X-axis by {steps} steps")
        except Exception as e:
            print("Error:", e)
    
    
    def move_y_by(self, steps):
        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            distance = steps * self.scale  # Change this value according to zooming
            feedrate = 450  # Feedrate in mm/minute
            gcode = f"G91\nG1 Y{distance} F{feedrate}\nG90\n"
            self.ser.write(gcode.encode())
            time.sleep(abs(distance)*0.15)
            self.ser.flush()
            status = self.is_busy()
            while "Run" in status :
                status = self.is_busy()
                print (status, "still running")
            self.y_pos = self.y_pos + steps
            print(f"Moved Y-axis by {steps} steps")
        except Exception as e:
            print("Error:", e)
    
    
    def move_xy_by(self, stepsX, stepsY):
        
        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            distanceX = stepsX * self.scale  
            distanceY = stepsY * self.scale
            feedrate = 450  # Feedrate in mm/minute
            gcode = f"G91\nG1 X{distanceX} Y{distanceY} F{feedrate}\nG90\n"
            self.ser.write(gcode.encode())
            time.sleep(max(abs(distanceX),abs(distanceY))*0.15)
            self.ser.flush()
            status = self.is_busy()
            while "Run" in status :
                status = self.is_busy()
                print (status, "still running")
            
            self.x_pos = self.x_pos + stepsX
            self.y_pos = self.y_pos + stepsY
            print(f"Moved X and Y by {stepsX},{stepsY} steps")
        except Exception as e:
            print("Error:", e) 
    
    
    def move_xy_to(self, x_pos, y_pos):
        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            feedrate = 450  # Feedrate in mm/minute
            gcode = f"G90\nG1 X{x_pos} Y{y_pos} Y0 F{feedrate}\n"
            self.ser.write(gcode.encode())
            status = self.is_busy()
            while "Run" in status :
                status = self.is_busy()
                print (status, "still running")
            
            self.x_pos = x_pos
            self.y_pos = y_pos
            print(f"Moved X and Y by {x_pos},{y_pos} steps")
        except Exception as e:
            print("Error:", e)
            
            
    def move_with_lim_by(self, stepsX, stepsY):
        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            current_x, current_y, current_z = self.get_current_position()
    
            if current_x is None or current_y is None:
                print("Error: Unable to determine current position")
                return 0
    
            distanceX = stepsX * self.scale
            distanceY = stepsY * self.scale
    
            final_x = current_x + distanceX
            final_y = current_y + distanceY
    
            if not (-37.5 <= final_x <= 37.5 and -37.5 <= final_y <= 37.5):
                print(f"Error: Movement exceeds limits. Current position: ({current_x}, {current_y}), Requested move: ({stepsX}, {stepsY})")
                return -1
    
            feedrate = 450  # Feedrate in mm/minute
            gcode = f"G91\nG1 X{distanceX} Y{distanceY} F{feedrate}\nG90\n"
            self.ser.write(gcode.encode())
            time.sleep(max(abs(distanceX), abs(distanceY)) * 0.15)
            
            status = self.is_busy()
            while "Run" in status:
                status = self.is_busy()
                print(status, "still running")
    
            print(f"Moved X and Y by {stepsX}, {stepsY} steps")
        except Exception as e:
            print("Error:", e)
        
        
    def move_z_by(self, height, feedrate=450):
        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            distance = height * self.scale  # Change this value according to the zooming
            gcode = f"G91\nG1 Z{distance} F{feedrate}\nG90\n"
            self.ser.write(gcode.encode())
            
            time.sleep(abs(distance)*0.15)
            self.ser.flush()
            status = self.is_busy()
            while "Run" in status :
                status = self.is_busy()
                print (status, "still running")
            self.z_pos = self.z_pos + height
            print(f"Moved Z-axis by {height} steps")
        except Exception as e:
            print("Error:", e)
            
            
    def move_z_infinite(self, max_height=300, feedrate=450):
        
        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            distance = max_height * self.scale  # Change this value according to the zooming
            gcode = f"G91\nG1 Z{distance} F{feedrate}\nG90\n"
            self.ser.write(gcode.encode())
        
        except Exception as e:
                print("Error:", e) 

    def abort_motion(self):
        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            pos = self.get_current_position()  # Get current position before abort
            
            # Send the hold command to stop motion
            gcode_hold = "!\n"
            self.ser.write(gcode_hold.encode())
            self.ser.flush()
            
            # Check current status and position
            gcode_status = "?\n"  # Status query command to get current position
            self.ser.write(gcode_status.encode())
            self.ser.flush()
    
            # Read the status and position after hold
            status = self.ser.readlines()
            print("Motion was aborted at position:", pos, "\n", "Current status:", status)
            
            # Clear the hold state without resuming motion
            # Option: send a zero-move command (G1 Z0) or other safe command to clear hold
            gcode_clear_hold = "G90 G1 Z0 F100\n"  # Absolute move to current position (no movement)
            self.ser.write(gcode_clear_hold.encode())
            self.ser.flush()
            
            print("Hold cleared without resuming previous motion")
            
        except Exception as e:
            print("Error:", e)


    def speak_grbl(self, gcode):        
        #for debugging or sending specialized GCodes.
        
        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            self.ser.write(gcode.encode())
            time.sleep(2)
            status = self.ser.readlines()
            print("Machine says:", status)
        except Exception as e:
                print("Error:", e)

        except Exception as e:
            print("Error:", e)
    
    def disconnect(self):
        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            self.go_to_zero()
            self.ser.close()
            print("Arduino disconnected")
        except Exception as e:
                print("Error:", e)

        except Exception as e:
            print("Error:", e)

    
class Light:

    def __init__(self):
        """Connect to the lights on the specified serial port."""
        
        try:
            self.ser_lights = serial.Serial("COM5", 9600, timeout=1)
            time.sleep(2)  # Wait for the connection to establish
            print("Connected to lights")
            
        except serial.SerialException as e:
            print(f"Error connecting to lights on: {e}")
            self.ser_lights = None

    def control_lights(self, command):
        """Send the command to turn lights on or off."""
        if self.ser_lights is not None:
            try:
                self.ser_lights.write(f"{command}\n".encode())  # Send command to the lights
                time.sleep(0.5)  # Wait for the Arduino to process the command
                print(f"Command '{command}' sent to lights.")
                
            except serial.SerialException as e:
                print(f"Error sending command '{command}': {e}")
        else:
            print("Error: Not connected to lights. Use 'connect_to_lights()' first.")
            
    def disconnect(self):
        self.ser_lights.write("off".encode())  # Send command to the lights
        time.sleep(0.5)
        self.ser_lights.close()
        print("lights are disconnectd")
            
class Zaber():
    
    def __init__(self, port):
        
        self.port = port
        self.connection = Connection.open_serial_port(port)
        self.device_list = self.connection.detect_devices()
        print("Found {} devices".format(len(self.device_list)))

        
        return None
    
    def get_devices(self):
        
        print(self.device_list)
        #assigning the separate arms to separate devices
        xdevice = self.device_list [1]
        ydevice = self.device_list[0]
        
        self.x1 = xdevice.get_axis(1)
        self.y1 = ydevice.get_axis(1)
       
        return self.x1, self.y1
            
    def disconnect(self):
        
        self.connection.close()
        print("Connection to Zaber stages is now closed")