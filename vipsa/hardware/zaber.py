"""
Zaber controller helpers for the ViPSA motion system.

IMPORTANT:

1. This code assumes that the Zaber stages are connected in a specific order:
   X stage on device 1 and Y stage on device 0. Verify the printed device list
   after detection and adjust `get_devices()` if the hardware order changes.
2. The default serial port for the current ViPSA setup is `COM7`.
3. When using native controller units, the stage resolution is approximately
   0.1 um per step. Pass `Units.*` constants to work in physical units instead.
"""

from zaber_motion import Units
from zaber_motion.ascii import Connection


class Zaber:
    """
    Initialization helper for Zaber X-LSQ150A stages.

    Once initialized, the returned axis objects can be used directly for
    movement commands and position queries.
    """

    def __init__(
        self,
        port: str = "COM7",
        distance_units=None,
        velocity_units=None,
        acceleration_units=None,
    ) -> None:
        """
        Connect to the Zaber stages and detect connected devices.

        By default, all values are sent in the controller's native units.
        Pass Zaber `Units.*` constants to use physical units instead, for
        example `Units.LENGTH_MILLIMETRES`.
        """

        self.port = port
        self.distance_units = distance_units
        self.velocity_units = velocity_units
        self.acceleration_units = acceleration_units

        self.connection = Connection.open_serial_port(port)
        self.device_list = self.connection.detect_devices()
        print("Found {} devices".format(len(self.device_list)))

        self.x1 = None
        self.y1 = None

    def get_devices(self) -> tuple:
        """
        Return the X and Y axes from the detected device list.
        """

        if len(self.device_list) < 2:
            raise RuntimeError(
                "Expected at least 2 Zaber devices, "
                f"but found {len(self.device_list)}."
            )

        print(self.device_list)
        xdevice = self.device_list[1]
        ydevice = self.device_list[0]

        self.x1 = xdevice.get_axis(1)
        self.y1 = ydevice.get_axis(1)

        self._prepare_axis(self.x1, "X")
        self._prepare_axis(self.y1, "Y")

        return self.x1, self.y1

    @staticmethod
    def _try_optional_axis_command(description: str, command) -> bool:
        """
        Run an optional Zaber setup command without failing older firmware.
        """

        try:
            command()
            return True
        except Exception as exc:
            print(f"Warning: skipped Zaber {description}: {exc}")
            return False

    def _prepare_axis(self, axis, axis_label: str) -> None:
        """
        Enable and unpark axes when supported by the connected controller.

        Some X-LSQ150A firmware rejects `driver enable`, while movement still
        works as in the legacy ViPSA driver. Treat these setup calls as optional.
        """

        self._try_optional_axis_command(
            f"{axis_label} driver enable",
            lambda: axis.driver_enable(),
        )

        try:
            is_parked = axis.is_parked()
        except Exception as exc:
            print(f"Warning: could not read Zaber {axis_label} park state: {exc}")
            return

        if is_parked:
            self._try_optional_axis_command(
                f"{axis_label} unpark",
                lambda: axis.unpark(),
            )

    def configure_units(
        self,
        distance_units=None,
        velocity_units=None,
        acceleration_units=None,
    ) -> None:
        """
        Update the default units used by movement and settings methods.

        Use `None` for native units.
        """

        self.distance_units = distance_units
        self.velocity_units = velocity_units
        self.acceleration_units = acceleration_units

    def _require_axis(self, axis_name: str):
        """Return the cached axis object, raising a helpful error if needed."""

        axis = getattr(self, axis_name, None)
        if axis is None:
            raise RuntimeError(
                "Zaber axes are not initialized. Call 'get_devices()' first."
            )
        return axis

    @staticmethod
    def _move(axis, method_name: str, value: float, units=None) -> None:
        """Call a Zaber move method with either native or explicit units."""

        move_method = getattr(axis, method_name)
        if units is None:
            move_method(value, Units.NATIVE)
            return
        move_method(value, units)

    @staticmethod
    def _set_axis_setting(axis, setting_name: str, value: float, units=None) -> None:
        """Set an axis setting in native units or explicit physical units."""

        if units is None:
            axis.settings.set(setting_name, value)
            return
        axis.settings.set(setting_name, value, units)

    @staticmethod
    def _get_axis_position(axis, units=None) -> float:
        """Read an axis position in native units or explicit physical units."""

        if units is None:
            return axis.get_position(Units.NATIVE)
        return axis.get_position(units)

    def set_holding_current(self, current: int = 50) -> None:
        """
        Set the holding current for both X and Y stages.

        Args:
            current: Holding current percentage from 0 to 100.
        """

        if not 0 <= current <= 100:
            raise ValueError("current must be between 0 and 100")

        x_axis = self._require_axis("x1")
        y_axis = self._require_axis("y1")

        x_axis.settings.set("driver.current.hold", current)
        y_axis.settings.set("driver.current.hold", current)

    def move_x_by(self, distance: float, units=None) -> None:
        """
        Move the X stage by a relative distance.

        If `units` is omitted, the class default is used. When the class
        default is `None`, the stage expects native units.
        """

        axis = self._require_axis("x1")
        resolved_units = self.distance_units if units is None else units
        self._move(axis, "move_relative", distance, resolved_units)

    def move_y_by(self, distance: float, units=None) -> None:
        """Move the Y stage by a relative distance."""

        axis = self._require_axis("y1")
        resolved_units = self.distance_units if units is None else units
        self._move(axis, "move_relative", distance, resolved_units)

    def move_x_to(self, position: float, units=None) -> None:
        """Move the X stage to an absolute position."""

        axis = self._require_axis("x1")
        resolved_units = self.distance_units if units is None else units
        self._move(axis, "move_absolute", position, resolved_units)

    def move_y_to(self, position: float, units=None) -> None:
        """Move the Y stage to an absolute position."""

        axis = self._require_axis("y1")
        resolved_units = self.distance_units if units is None else units
        self._move(axis, "move_absolute", position, resolved_units)

    def get_current_position(self, units=None) -> tuple:
        """
        Return the current X and Y positions.

        If `units` is omitted, the class distance default is used. When the
        class default is `None`, the controller returns native units.
        """

        x_axis = self._require_axis("x1")
        y_axis = self._require_axis("y1")
        resolved_units = self.distance_units if units is None else units

        x_pos = self._get_axis_position(x_axis, resolved_units)
        y_pos = self._get_axis_position(y_axis, resolved_units)
        return x_pos, y_pos

    def set_maxspeed_x(self, speed: float, units=None) -> None:
        """Set the X-axis maximum speed."""

        axis = self._require_axis("x1")
        resolved_units = self.velocity_units if units is None else units
        self._set_axis_setting(axis, "maxspeed", speed, resolved_units)

    def set_maxspeed_y(self, speed: float, units=None) -> None:
        """Set the Y-axis maximum speed."""

        axis = self._require_axis("y1")
        resolved_units = self.velocity_units if units is None else units
        self._set_axis_setting(axis, "maxspeed", speed, resolved_units)

    def set_acceleration_x(self, acceleration: float, units=None) -> None:
        """Set the X-axis acceleration."""

        axis = self._require_axis("x1")
        resolved_units = self.acceleration_units if units is None else units
        self._set_axis_setting(
            axis,
            "accel",
            acceleration,
            resolved_units,
        )

    def set_acceleration_y(self, acceleration: float, units=None) -> None:
        """Set the Y-axis acceleration."""

        axis = self._require_axis("y1")
        resolved_units = self.acceleration_units if units is None else units
        self._set_axis_setting(
            axis,
            "accel",
            acceleration,
            resolved_units,
        )

    def get_default_units(self) -> dict:
        """Return the currently configured default units for the controller."""

        return {
            "distance_units": self.distance_units,
            "velocity_units": self.velocity_units,
            "acceleration_units": self.acceleration_units,
        }

    def disconnect(self) -> None:
        """
        Disconnect from the Zaber stages by closing the connection.
        """

        for axis_name in ("x1", "y1"):
            axis = getattr(self, axis_name, None)
            if axis is None:
                continue

            try:
                is_parked = axis.is_parked()
            except Exception as exc:
                print(f"Warning: could not read Zaber {axis_name} park state: {exc}")
                is_parked = True

            if not is_parked:
                self._try_optional_axis_command(
                    f"{axis_name} park",
                    lambda: axis.park(),
                )

            self._try_optional_axis_command(
                f"{axis_name} driver disable",
                lambda: axis.driver_disable(),
            )

        self.connection.close()
        print("Connection to Zaber stages is now closed")


__all__ = ["Zaber", "Units"]
