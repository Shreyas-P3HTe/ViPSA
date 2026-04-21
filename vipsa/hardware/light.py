"""
Light controller for the ViPSA top illumination.

The default configuration assumes an Adafruit NeoPixel strip controlled 
by an Arduino. The Arduino should be running firmware that listens for 
simple text commands over the serial port to control the light strip.

Additionally, the the adfruit neopixel library should be imported in the arduino script.

As of this writing, change_intensity() is a placeholder for future support of intensity control, 
which is not yet implemented in the Arduino firmware.

Authors
------- 
    Shreyas Pethe 2026/04/20
"""

import time
from typing import Optional

import serial


class Light:
    """
    Control the top light for the micropositioning system.

    The light is an Adafruit NeoPixel strip controlled by an Arduino over a
    serial connection.
    """

    VALID_COMMANDS = {"on", "off", "green", "red", "rainbow"}

    def __init__(
        self,
        port: str = "COM6",
        baudrate: int = 9600,
        timeout: float = 1,
        connection_delay: float = 2.0,
        command_delay: float = 0.5,
    ) -> None:
        """
        Configure the light controller.

        This does not open the serial connection. Call `connect()` explicitly
        when you want to talk to the hardware.
        """

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection_delay = connection_delay
        self.command_delay = command_delay
        self.ser_lights: Optional[serial.Serial] = None

    @property
    def is_connected(self) -> bool:
        """Return True when the serial connection is open."""

        return self.ser_lights is not None and self.ser_lights.is_open

    def connect(self) -> bool:
        """
        Open the serial connection to the light controller.

        Returns:
            bool: True when the connection is available, False otherwise.
        """

        if self.is_connected:
            return True

        try:
            self.ser_lights = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout,
            )
            time.sleep(self.connection_delay)
            print(f"Connected to lights on {self.port}")
            return True
        except serial.SerialException as exc:
            print(f"Error connecting to lights on {self.port}: {exc}")
            self.ser_lights = None
            return False

    def initialize(self, startup_command: Optional[str] = "off") -> bool:
        """
        Initialize the controller after connecting.

        Args:
            startup_command: Optional command to send once the serial link is
                ready. Use `None` to skip sending a startup command.

        Returns:
            bool: True when initialization succeeds.
        """

        if not self.is_connected:
            print("Error: Not connected to lights. Call 'connect()' first.")
            return False

        if startup_command is None:
            return True

        return self.control_lights(startup_command)

    def control_lights(self, command: str) -> bool:
        """
        Send a command to the light controller.

        Valid commands include:
            "on" - turn the lights on
            "off" - turn the lights off
            "green" - set the lights to green
            "red" - set the lights to red
            "rainbow" - set the lights to rainbow mode
        """

        if command not in self.VALID_COMMANDS:
            print(
                f"Error: '{command}' is not a valid light command. "
                f"Valid commands: {sorted(self.VALID_COMMANDS)}"
            )
            return False

        if not self.is_connected:
            print("Error: Not connected to lights. Call 'connect()' first.")
            return False

        assert self.ser_lights is not None

        try:
            self.ser_lights.write(f"{command}\n".encode("utf-8"))
            self.ser_lights.flush()
            time.sleep(self.command_delay)
            print(f"Command '{command}' sent to lights.")
            return True
        except serial.SerialException as exc:
            print(f"Error sending command '{command}': {exc}")
            return False

    def turn_on(self) -> bool:
        """Turn the lights on."""

        return self.control_lights("on")

    def turn_off(self) -> bool:
        """Turn the lights off."""

        return self.control_lights("off")

    def set_green(self) -> bool:
        """Set the lights to green."""

        return self.control_lights("green")

    def set_red(self) -> bool:
        """Set the lights to red."""

        return self.control_lights("red")

    def set_rainbow(self) -> bool:
        """Set the lights to rainbow mode."""

        return self.control_lights("rainbow")

    def change_intensity(self, percentage: int) -> None:
        """
        Placeholder for future intensity control support.

        NOTE:
            Intensity control is not yet implemented in the Arduino firmware.
        """

        if not 0 <= percentage <= 100:
            raise ValueError("percentage must be between 0 and 100")

    def disconnect(self, turn_off_first: bool = True) -> None:
        """
        Disconnect from the lights.

        Args:
            turn_off_first: When True, send an "off" command before closing the
                serial connection.
        """

        if not self.is_connected:
            self.ser_lights = None
            return

        try:
            if turn_off_first:
                self.control_lights("off")
            if self.ser_lights is not None:
                self.ser_lights.close()
            print("Lights disconnected")
        except serial.SerialException as exc:
            print(f"Error disconnecting lights: {exc}")
        finally:
            self.ser_lights = None
