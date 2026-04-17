# -*- coding: utf-8 -*-
"""
Shreyas' code.

Module for controlling movement related hardware on ViPSA

Classes:

    Stage()
        > Openflexture stage runs on grbl flashed on arduino UNO. (https://openflexure.org/projects/blockstage)
        > The hardware comprises of a CNC shield for arduino and 3 pololu DRV8834 stepper motor drivers.
        > These drivers control 3 stepper motors(28BYJ-48, 5V).
        > There are no limit switches at the end of these motors and thus they are "blind".
        > The stage position is tracked by initializing a variable of for each of the three axes in 
            the beginning of a session and then upating them each time any movement is done  
        
    Light()
        > This class controls the top light for the micropositioning system.
        > The light is an adafruit neopixel strip of 64 LEDs, controlled through an arduino uno       
        
    Zaber()
        Helper class for initialization of the zaber stages (Zaber X-LSQ150A)
        Once initialized, the separate objects for zaber stages can directly use methods from the 
"""

import serial
import time
from typing import Optional, Tuple
from zaber_motion import Library, Units
from zaber_motion.ascii import Connection, AllAxes, Axis, Device

class Stage():
    """
    The class for controlling the openflexture stage.
    Sometimes simply referred to as "Arduino stage"

    G-code is passed to the GRBL firmware flashed on the arduino,
    which then results in the motion via the stepper motors

    External dependancies : serial.
    """

    def __init__(self, port:str, baudrate:int, multiplier:float) -> None:

        """
        Args :
            port - COM port of the stage arduino.
            baudrate - communication rate. typically 115200.
            multiplier - this is useful for setups that use different optics. Will be fixed on ViPSA.
        
        Returns :
            None
        
        """

        self.scale = multiplier                             #this changs according to the zoom level of the optics. 
        self.port = port
        self.baudrate = baudrate

        self.x_pos = 0                                          #position variables for tracking
        self.y_pos = 0  
        self.z_pos = 0
        
        try:
            self.ser = serial.Serial(port, baudrate, timeout=2)
            time.sleep(2)  
            print("Connected to Arduino")
        except Exception as e:
            print("Error:", e)
            self.ser = None
    
    def flush(self, clear_input=True, clear_output=False) -> bool:

        """
        Synchronize the serial link and optionally clear stale buffered data.

        Note:
            ``pyserial.flush()`` only waits for pending writes to finish. It
            does not clear unread incoming bytes. This helper drains outgoing
            commands first, then clears buffers when requested.
        """

        if self.ser is None or not self.ser.is_open:
            print("Arduino not connected")
            return False
        
        try:
            self.ser.flush()

            if clear_input:
                self.ser.reset_input_buffer()

            if clear_output:
                self.ser.reset_output_buffer()

            print(
                f"Serial synchronized"
                f" (input_cleared={clear_input}, output_cleared={clear_output})"
            )
            return True

        except serial.SerialException as e:
            print("Serial flush failed:", e)
            return False
                        
    def is_busy(self) -> Optional[str] : 
        
        """Query the GRBL status to check if the stage is currently executing a motion command."""

        if self.ser is None or not self.ser.is_open:
            print("Arduino not connected")
            return None
        
        try :
            # Send status query
            time.sleep(0.3)
            self.ser.write(b'?')
            time.sleep(0.3)
            status = self.ser.readline().decode('utf-8').strip()
            return status
            
        except Exception as e:
            print("Error:", e)
            
    def _change_scale(self, new_scale) -> None:
        """Change the scale factor for movement commands, 
        which is useful for different zoom levels of the optics.
        Not used currently but can be useful for future iterations of the system."""

        old_scale = self.scale
        self.scale = new_scale
        print(f"Scaling changed from {old_scale} to {self.scale}")
                   
    def set_zero(self) -> None:
        """Set the current position of the stage as the new zero reference point.
        This is useful for initializing the stage position at the beginning of a 
        session or after a manual adjustment.
        
        Note: This does not move the stage physically, 
        it just resets the internal position tracking variables to zero."""

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
    
    def go_to_zero(self) -> None:
        """Move the stage to the zero position.
        Note: This will physically move the stage to the zero reference point.
        """

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
    
    def get_current_position(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Query the current position of the stage from the GRBL firmware.
        Returns a tuple of (x, y, z) positions in millimeters, or
        (None, None, None) if there was an error.
        """

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return None, None, None

        try:
            x = self.x_pos
            y = self.y_pos
            z = self.z_pos
            return x, y, z
        except Exception as e:
            print("Error querying or parsing position:", e)
            return None, None, None
        
    def move_x_by(self, steps: float) -> None:
        """ Move the stage along the X-axis by a specified number of steps."""

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
            
            while status is not None and "Run" in status :
                status = self.is_busy()
                print (status, "still running")
            
            self.x_pos = self.x_pos + steps 
            print(f"Moved X-axis by {steps} steps")
        except Exception as e:
            print("Error:", e)
        
    def move_y_by(self, steps: float) -> None:
        """ Move the stage along the Y-axis by a specified number of steps."""

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
            while status is not None and "Run" in status :
                status = self.is_busy()
                print (status, "still running")
            self.y_pos = self.y_pos + steps
            print(f"Moved Y-axis by {steps} steps")
        except Exception as e:
            print("Error:", e)
        
    def move_xy_by(self, stepsX: float, stepsY: float) -> None:
        """ Move the stage along both X and Y axes by specified numbers of steps.
        This method allows for simultaneous movement along both axes, 
        which can be more efficient than moving them sequentially."""

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
            while status is not None and "Run" in status :
                status = self.is_busy()
                print (status, "still running")
            
            self.x_pos = self.x_pos + stepsX
            self.y_pos = self.y_pos + stepsY
            print(f"Moved X and Y by {stepsX},{stepsY} steps")
        except Exception as e:
            print("Error:", e) 
        
    def move_xy_to(self, x_pos: float, y_pos: float) -> None:
        """ Move the stage to specific X and Y coordinates (in steps).
        This method calculates the required movement from the current position to the target coordinates"""

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            feedrate = 450  # Feedrate in mm/minute
            gcode = f"G90\nG1 X{x_pos} Y{y_pos} Y0 F{feedrate}\n"
            self.ser.write(gcode.encode())
            status = self.is_busy()
            while status is not None and "Run" in status :
                status = self.is_busy()
                print (status, "still running")
            
            self.x_pos = x_pos
            self.y_pos = y_pos
            print(f"Moved X and Y by {x_pos},{y_pos} steps")
        except Exception as e:
            print("Error:", e)
                        
    def move_with_lim_by(self, stepsX: float, stepsY: float) -> int:
        """ Move the stage along both X and Y axes by specified numbers of steps,
            but with limits to prevent moving beyond the physical boundaries of the stage.
            This method checks the current position and the requested movement against predefined limits
            NOTE : The limits are currently set to -37.5 to 37.5 steps for both X and Y axes, which corresponds to the physical range of the stage.
            """

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return 0
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
            while status is not None and "Run" in status:
                status = self.is_busy()
                print(status, "still running")
    
            print(f"Moved X and Y by {stepsX}, {stepsY} steps")
        except Exception as e:
            print("Error:", e)
                
    def move_z_by(self, height: float, feedrate: int = 450) -> None:
        """ Move the stage along the Z-axis by a specified height (in steps).
            This method allows for movement along the Z-axis, 
            which is heavily used for probing.
            NOte : The feedrate is exposed and can be adjusted for faster or slower movement along the Z-axis."""

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
            while status is not None and "Run" in status :
                status = self.is_busy()
                print (status, "still running")
            self.z_pos = self.z_pos + height
            print(f"Moved Z-axis by {height} steps")
        except Exception as e:
            print("Error:", e)
                        
    def _move_z_infinite(self, max_height: float = 300, feedrate: int = 450) -> None:
        """ Move the stage along the Z-axis infinitely until stopped.
            This method is useful for continuous scanning or probing operations.
            DEPRECATED : This method is not currently used and may be removed in future iterations, but can be useful for certain applications that require continuous movement along the Z-axis.
        """

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            distance = max_height 
            gcode = f"G91\nG1 Z{distance} F{feedrate}\nG90\n"
            self.ser.write(gcode.encode())
        
        except Exception as e:
                print("Error:", e) 

    def abort_motion(self) -> None:
        """Abort the current motion immediately without resuming previous commands.
        This method sends a hold command to the GRBL firmware to stop all motion immediately.
        """

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

    def reconnect(self) -> None:
        """Reconnect to the Arduino stage if the connection was lost.
        This method attempts to close any existing serial connection and then re-establishes a new connection to the Arduino. It also preserves the last known position of the stage to avoid losing track of the current location after reconnection.
        """

        previous_position = (self.x_pos, self.y_pos, self.z_pos)
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2)
            self.x_pos, self.y_pos, self.z_pos = previous_position
            print("Arduino reconnected")
        except Exception as e:
            self.ser = None
            print("Error reconnecting Arduino:", e)

    def _speak_grbl(self, gcode:str) -> None:

        """Send a raw G-code command directly to the GRBL firmware on the Arduino.
        This method allows for sending custom G-code commands that may not be covered by the existing movement methods. It can be useful for advanced users who want to execute specific G-code commands for specialized operations or debugging purposes. Note that using this method requires knowledge of G-code and the GRBL command set, and improper use can lead to unintended behavior of the stage, so it should be used with caution."""

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
    
    def disconnect(self) -> None:
        """Disconnect from the Arduino stage and move to the zero position before closing the connection."""

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

    def __init__(self) -> None:
        """Connect to the lights on the specified serial port."""
        
        try:
            self.ser_lights = serial.Serial("COM6", 9600, timeout=1)
            time.sleep(2)  # Wait for the connection to establish
            print("Connected to lights")
            
        except serial.SerialException as e:
            print(f"Error connecting to lights on: {e}")
            self.ser_lights = None

    def control_lights(self, command: str) -> None:
        """Send the command to turn lights on or off.
        
        Args:
            command (str): The command to send to the lights.

        Note : The string commands are defined in the arduino code for the 
        lights and should be consistent with that.

        Valid commands include:         
                "on" - to turn the lights on
                "off" - to turn the lights off
                "green" - to set the lights to green
                "red" - to set the lights to red
                "rainbow" - to set the lights to rainbow mode
        """

        if self.ser_lights is not None:
            try:
                self.ser_lights.write(f"{command}\n".encode())  # Send command to the lights
                time.sleep(0.5)  # Wait for the Arduino to process the command
                print(f"Command '{command}' sent to lights.")
                
            except serial.SerialException as e:
                print(f"Error sending command '{command}': {e}")
        else:
            print("Error: Not connected to lights. Use 'connect_to_lights()' first.")
            
    def disconnect(self) -> None:
        """Disconnect from the lights and turn them off before closing the connection."""

        self.ser_lights.write("off".encode())  # Send command to the lights
        time.sleep(0.5)
        self.ser_lights.close()
        print("lights are disconnectd")
            
class Zaber():

    """Currently a lazily implemented class for initialization of the zaber stages (Zaber X-LSQ150A).

    TO DO FOR PUDA :
        Add helper methods for moving the zaber stages and getting their positions, 
        e.g. move_x_by, move_y_by, move_xy_by, move_xy_to, get_current_position, etc.
        similar to the Stage class."""
    
    def __init__(self, port: str) -> None:
        """Connect to the Zaber stages on the specified serial port and detect connected devices.
        The Connection class from the zaber_motion library is used to establish a connection to the 
        Zaber stages via the specified serial port. 
        This connection object will be used for all subsequent communication with the stages.
        """

        self.port = port
        self.connection = Connection.open_serial_port(port) 
        self.device_list = self.connection.detect_devices()
        print("Found {} devices".format(len(self.device_list)))
        
        return None
    
    def get_devices(self) -> tuple:
        """Get the X and Y devices from the detected device list and return their 
        corresponding axes for control.
        Note: The device list is expected to contain the X and Y stages, 
        and this method assigns them based on their order in the list. 
        It returns the axis objects for the X and Y stages, 
        which can then be used for movement commands and position queries."""

        print(self.device_list)
        #assigning the separate arms to separate devices
        xdevice = self.device_list [1]
        ydevice = self.device_list[0]
        
        self.x1 = xdevice.get_axis(1)
        self.y1 = ydevice.get_axis(1)
       
        return self.x1, self.y1
            
    def disconnect(self) -> None:
        """Disconnect from the Zaber stages by closing the connection.
        There is no homing or zeroing involved in this process.
        Zaber stages can be left at their current position when disconnecting, as they have encoders.
        """

        self.connection.close()
        print("Connection to Zaber stages is now closed")
