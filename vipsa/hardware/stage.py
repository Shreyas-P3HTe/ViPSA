"""
Stage controller for the ViPSA OpenFlexure-based Arduino stage.
"""

import time
from typing import Optional, Tuple

import serial


class Stage:
    """
    Control the OpenFlexure stage.

    G-code is passed to the GRBL firmware flashed on the Arduino, which then
    drives the stepper motors (28BYJ-48). The motor drivers used are Pololu drv8825,
    which support microstepping up to 1/32. 
    
    The stage is designed to have a maximum travel of approximately 2mm in all 3 axes.
    To track the current position, the class maintains internal variables that are updated after each movement command.
    This allows the `get_current_position()` method to return the latest position.

    """

    def __init__(self, port: str, baudrate: int, multiplier: float) -> None:
        """
        Args:
            port: COM port of the stage Arduino.
            baudrate: Communication rate, typically 115200.
            multiplier: Scale factor used for setups with different optics.
        """

        self.scale = multiplier
        self.port = port
        self.baudrate = baudrate

        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0

        try:
            self.ser = serial.Serial(port, baudrate, timeout=2)
            time.sleep(2)
            print("Connected to Arduino")
        except Exception as exc:
            print("Error:", exc)
            self.ser = None

    def flush(self, clear_input: bool = True, clear_output: bool = False) -> bool:
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

        except serial.SerialException as exc:
            print("Serial flush failed:", exc)
            return False

    def is_busy(self) -> Optional[str]:
        """Query GRBL status to check whether a motion command is still running."""

        if self.ser is None or not self.ser.is_open:
            print("Arduino not connected")
            return None

        try:
            time.sleep(0.3)
            self.ser.write(b"?")
            time.sleep(0.3)
            status = self.ser.readline().decode("utf-8").strip()
            return status
        except Exception as exc:
            print("Error:", exc)
            return None

    def _change_scale(self, new_scale: float) -> None:
        """
        Change the scale factor for movement commands.

        This is useful for different zoom levels of the optics.
        """

        old_scale = self.scale
        self.scale = new_scale
        print(f"Scaling changed from {old_scale} to {self.scale}")

    def set_zero(self) -> None:
        """
        Set the current position of the stage as the new zero reference point.

        This does not move the stage physically; it resets the internal
        position tracking variables.
        """

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
        except Exception as exc:
            print("Error:", exc)

    def go_to_zero(self) -> None:
        """Move the stage to the zero position."""

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
        except Exception as exc:
            print("Error:", exc)

    def get_current_position(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Return the tracked current stage position as (x, y, z).

        Returns `(None, None, None)` if the stage is not connected.
        """

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return None, None, None

        try:
            return self.x_pos, self.y_pos, self.z_pos
        except Exception as exc:
            print("Error querying or parsing position:", exc)
            return None, None, None

    def move_x_by(self, steps: float) -> None:
        """Move the stage along the X-axis by a specified number of steps."""

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            distance = steps * self.scale
            feedrate = 450
            gcode = f"G91\nG1 X{distance} F{feedrate}\nG90\n"
            self.ser.write(gcode.encode())
            time.sleep(abs(distance) * 0.15)
            self.ser.flush()
            status = self.is_busy()

            while status is not None and "Run" in status:
                status = self.is_busy()
                print(status, "still running")

            self.x_pos = self.x_pos + steps
            print(f"Moved X-axis by {steps} steps")
        except Exception as exc:
            print("Error:", exc)

    def move_y_by(self, steps: float) -> None:
        """Move the stage along the Y-axis by a specified number of steps."""

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            distance = steps * self.scale
            feedrate = 450
            gcode = f"G91\nG1 Y{distance} F{feedrate}\nG90\n"
            self.ser.write(gcode.encode())
            time.sleep(abs(distance) * 0.15)
            self.ser.flush()
            status = self.is_busy()
            while status is not None and "Run" in status:
                status = self.is_busy()
                print(status, "still running")
            self.y_pos = self.y_pos + steps
            print(f"Moved Y-axis by {steps} steps")
        except Exception as exc:
            print("Error:", exc)

    def move_xy_by(self, stepsX: float, stepsY: float) -> None:
        """
        Move the stage along both X and Y axes by specified numbers of steps.

        This allows simultaneous movement along both axes, which is more
        efficient than moving them sequentially.
        """

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            distanceX = stepsX * self.scale
            distanceY = stepsY * self.scale
            feedrate = 450
            gcode = f"G91\nG1 X{distanceX} Y{distanceY} F{feedrate}\nG90\n"
            self.ser.write(gcode.encode())
            time.sleep(max(abs(distanceX), abs(distanceY)) * 0.15)
            self.ser.flush()
            status = self.is_busy()
            while status is not None and "Run" in status:
                status = self.is_busy()
                print(status, "still running")

            self.x_pos = self.x_pos + stepsX
            self.y_pos = self.y_pos + stepsY
            print(f"Moved X and Y by {stepsX},{stepsY} steps")
        except Exception as exc:
            print("Error:", exc)

    def move_xy_to(self, x_pos: float, y_pos: float) -> None:
        """
        Move the stage to specific X and Y coordinates in steps.

        This calculates the required movement from the current position to the
        target coordinates.
        """

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            feedrate = 450
            gcode = f"G90\nG1 X{x_pos} Y{y_pos} Y0 F{feedrate}\n"
            self.ser.write(gcode.encode())
            status = self.is_busy()
            while status is not None and "Run" in status:
                status = self.is_busy()
                print(status, "still running")

            self.x_pos = x_pos
            self.y_pos = y_pos
            print(f"Moved X and Y by {x_pos},{y_pos} steps")
        except Exception as exc:
            print("Error:", exc)

    def move_with_lim_by(self, stepsX: float, stepsY: float) -> int:
        """
        Move within the stage boundaries.

        Returns:
            int: `-1` when the movement exceeds limits, `0` when not connected,
            and `None` on success to preserve existing behavior.
        """

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return 0
        try:
            current_x, current_y, _current_z = self.get_current_position()

            if current_x is None or current_y is None:
                print("Error: Unable to determine current position")
                return 0

            distanceX = stepsX * self.scale
            distanceY = stepsY * self.scale

            final_x = current_x + distanceX
            final_y = current_y + distanceY

            if not (-37.5 <= final_x <= 37.5 and -37.5 <= final_y <= 37.5):
                print(
                    "Error: Movement exceeds limits. "
                    f"Current position: ({current_x}, {current_y}), "
                    f"Requested move: ({stepsX}, {stepsY})"
                )
                return -1

            feedrate = 450
            gcode = f"G91\nG1 X{distanceX} Y{distanceY} F{feedrate}\nG90\n"
            self.ser.write(gcode.encode())
            time.sleep(max(abs(distanceX), abs(distanceY)) * 0.15)

            status = self.is_busy()
            while status is not None and "Run" in status:
                status = self.is_busy()
                print(status, "still running")

            print(f"Moved X and Y by {stepsX}, {stepsY} steps")
        except Exception as exc:
            print("Error:", exc)

    def move_z_by(self, height: float, feedrate: int = 450) -> None:
        """
        Move the stage along the Z-axis by a specified height in steps.
        """

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            distance = height * self.scale
            gcode = f"G91\nG1 Z{distance} F{feedrate}\nG90\n"
            self.ser.write(gcode.encode())

            time.sleep(abs(distance) * 0.15)
            self.ser.flush()
            status = self.is_busy()
            while status is not None and "Run" in status:
                status = self.is_busy()
                print(status, "still running")
            self.z_pos = self.z_pos + height
            print(f"Moved Z-axis by {height} steps")
        except Exception as exc:
            print("Error:", exc)

    def _move_z_infinite(self, max_height: float = 300, feedrate: int = 450) -> None:
        """
        Move the stage along the Z-axis continuously until externally stopped.
        """

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            gcode = f"G91\nG1 Z{max_height} F{feedrate}\nG90\n"
            self.ser.write(gcode.encode())
        except Exception as exc:
            print("Error:", exc)

    def abort_motion(self) -> None:
        """
        Abort the current motion immediately without resuming previous commands.
        """

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            pos = self.get_current_position()

            gcode_hold = "!\n"
            self.ser.write(gcode_hold.encode())
            self.ser.flush()

            gcode_status = "?\n"
            self.ser.write(gcode_status.encode())
            self.ser.flush()

            status = self.ser.readlines()
            print("Motion was aborted at position:", pos, "\n", "Current status:", status)

            gcode_clear_hold = "G90 G1 Z0 F100\n"
            self.ser.write(gcode_clear_hold.encode())
            self.ser.flush()

            print("Hold cleared without resuming previous motion")
        except Exception as exc:
            print("Error:", exc)

    def reconnect(self) -> None:
        """
        Reconnect to the Arduino stage if the connection was lost.
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
        except Exception as exc:
            self.ser = None
            print("Error reconnecting Arduino:", exc)

    def _speak_grbl(self, gcode: str) -> None:
        """
        Send a raw G-code command directly to the GRBL firmware on the Arduino.
        """

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            self.ser.write(gcode.encode())
            time.sleep(2)
            status = self.ser.readlines()
            print("Machine says:", status)
        except Exception as exc:
            print("Error:", exc)

    def disconnect(self) -> None:
        """Disconnect from the Arduino stage after moving to zero."""

        if self.ser is None or not self.ser.is_open:
            print("Error: Arduino not connected")
            return
        try:
            self.go_to_zero()
            self.ser.close()
            print("Arduino disconnected")
        except Exception as exc:
            print("Error:", exc)


stage = Stage

__all__ = ["Stage", "stage"]
