"""PyVISA wrapper for the Keithley 707B switch matrix.

This module is routing-only. It exposes named routes and channel helpers for
the 707B matrix and intentionally contains no source/measure or protocol logic.
"""

# TODO: Resolve remaining legacy compatibility gaps while verifying channel
# routing commands on a real 707B.

from __future__ import annotations

from typing import Any

import pyvisa


class Keithley707B:
	"""Minimal Keithley 707B switch wrapper.

	The class keeps a persistent VISA session and exposes a small set of route
	helpers that are convenient for switching between named SMU paths. It is a
	switch-routing layer only and does not perform measurement operations.

	Attributes:
		address:
			VISA resource address of the 707B.
		slot:
			Default slot used when building channel identifiers.
		route_map:
			Named route lookup table used by ``connect_named_route``.
	"""

	DEFAULT_SLOT = 1
	ROUTE_MAP = {
		"keithley": (("A", 1), ("B", 3)),
		"keysight": (("A", 2), ("B", 4)),
	}

	def __init__(
		self,
		device_no: int = 0,
		address: str | None = None,
		slot: int = DEFAULT_SLOT,
	) -> None:
		"""Initialize the 707B wrapper and open no session yet.

		Args:
			device_no:
				Index into the VISA resource list when an explicit address is not
				provided.
			address:
				Optional explicit VISA resource address.
			slot:
				Default switch-card slot used when channel tokens are generated.
		"""
		self.rm = pyvisa.ResourceManager()
		address_list = list(self.rm.list_resources())
		self.address = address if address is not None else address_list[device_no]
		self.slot = int(slot)
		self.route_map = dict(self.ROUTE_MAP)
		self._dev = None

	def _open_device(self):
		"""Open the VISA session lazily and return it.

		The 707B session is opened on first use so callers can construct the
		wrapper without immediately touching hardware.
		"""
		if self._dev is None:
			dev = self.rm.open_resource(self.address)
			dev.timeout = 10000
			dev.write_termination = "\n"
			dev.read_termination = "\n"
			self._dev = dev
		return self._dev

	def _close_device(self) -> None:
		"""Close the internal VISA session if one is open."""
		try:
			if self._dev is not None:
				self._dev.close()
		finally:
			self._dev = None

	def _is_route_name(self, value: Any) -> bool:
		"""Return ``True`` when ``value`` names a known route.

		This lets public helpers accept either route names such as ``"keithley"``
		or explicit channel specifications.

		Args:
			value:
				Any value to check for being a route name.
		
		Returns:
			``True`` if the value is a known route name, else ``False``.
		"""
		return isinstance(value, str) and value.strip().lower() in self.route_map

	def build_channel(
		self,
		row: str,
		column: int,
		slot: int | None = None,
	) -> str:
		"""Build a 707B channel token such as ``1A03``.

		Args:
			row:
				Row identifier, typically ``A`` or ``B``.
			column:
				Column identifier.
			slot:
				Optional slot override. Defaults to ``self.slot``.

		Returns:
			A channel token formatted for the 707B.
		"""
		slot_value = self.slot if slot is None else int(slot)
		row_value = str(row).strip().upper()
		column_value = int(column)
		return f"{slot_value}{row_value}{column_value:02d}"

	def _normalize_channel(self, channel: Any) -> str:
		"""Normalize one channel specification into a 707B channel token.

		Flexible normalization is intentional for backward compatibility. This
		method accepts dicts, tuples, raw channel strings, and the special values
		``"all"`` / ``"allslots"``.

		Args:
			channel:
				Channel specification, which can be a dict with keys ``row``, ``column`
				and optionally ``slot``, a tuple of (row, column) or (slot, row, column),
				or a string such as "1A03" or "allslots".

		Returns:
			A normalized 707B channel token such as ``"1A03"`` or ``"allslots"``.
		"""
		if isinstance(channel, dict):
			return self.build_channel(
				row=channel["row"],
				column=channel["column"],
				slot=channel.get("slot", self.slot),
			)

		if isinstance(channel, tuple):
			if len(channel) == 2:
				return self.build_channel(channel[0], channel[1])
			if len(channel) == 3:
				return self.build_channel(channel[1], channel[2], slot=channel[0])

		if isinstance(channel, str):
			text = channel.strip()
			lowered = text.lower()
			if lowered in {"all", "allslots"}:
				return "allslots"
			return text.upper()

		return str(channel)

	def get_route_channels(
		self,
		route_name: str,
		slot: int | None = None,
	) -> list[str]:
		"""Return the channel list corresponding to a named route.

		Args:
			route_name:
				Route name such as ``"keithley"`` or ``"keysight"``.
			slot:
				Optional slot override for the route's channels.

		Returns:
			A list of normalized channel tokens for the requested route.
		"""
		route_key = str(route_name).strip().lower()
		if route_key not in self.route_map:
			raise ValueError(f"Unknown 707B route '{route_name}'.")
		return [
			self.build_channel(row=row, column=column, slot=slot)
			for row, column in self.route_map[route_key]
		]

	def _resolve_channels(self, route: Any) -> list[str] | str:
		"""Resolve a route name or channel specification into channel tokens."""
		if self._is_route_name(route):
			return self.get_route_channels(str(route))

		if isinstance(route, list):
			resolved: list[str] = []
			for item in route:
				item_resolved = self._resolve_channels(item)
				if isinstance(item_resolved, list):
					resolved.extend(item_resolved)
				else:
					resolved.append(item_resolved)
			return resolved

		return self._normalize_channel(route)

	def _format_channels(self, route: Any) -> str:
		"""Format one or more channels into the 707B comma-separated syntax."""
		channels = self._resolve_channels(route)
		if isinstance(channels, list):
			return ",".join(channels)
		return channels

	def send_switch_command(self, command: str) -> None:
		"""Send a raw switch command to the 707B.

		Args:
			command:
				TSP command string such as ``channel.open("1A03")``.
		"""
		try:
			self._open_device().write(command)
		except pyvisa.errors.VisaIOError:
			self._close_device()
			self._open_device().write(command)

	def query_switch_expression(self, expression: str) -> str:
		"""Evaluate one switch expression and return the printed result.

		Args:
			expression:
				TSP expression string to evaluate.

		Returns:
			The printed result of the expression.
		"""
		dev = self._open_device()
		dev.write(f"print({expression})")
		return dev.read().strip()

	def write_tsp(self, command: str) -> None:
		"""Backward-compatible alias for ``send_switch_command``.

		Args:
			command:
				Raw TSP command string.
		"""
		self.send_switch_command(command)

	def query_tsp(self, expression: str) -> str:
		"""Backward-compatible alias for ``query_switch_expression``.

		Args:
			expression:
				TSP expression string to evaluate.

		Returns:
			The printed result of the expression.
		"""
		return self.query_switch_expression(expression)

	def close_channel(self, channel: Any) -> None:
		"""Close one or more channels.

		Args:
			channel:
				Route name or channel specification to close.
		"""
		chs = self._format_channels(channel)
		self.send_switch_command(f'channel.close("{chs}")')

	def open_channel(self, channel: Any) -> None:
		"""Open one or more channels.

		Args:
			channel:
				Route name or channel specification to open.
		"""
		chs = self._format_channels(channel)
		self.send_switch_command(f'channel.open("{chs}")')

	def open_all(self) -> None:
		"""Open all channels in all slots."""
		self.send_switch_command('channel.open("allslots")')

	def get_closed_channels(self) -> str:
		"""Return the raw 707B report of closed channels."""
		return self.query_switch_expression('channel.getclose("allslots")')

	def connect_named_route(
		self,
		route_name: str,
		slot: int | None = None,
	) -> list[str]:
		"""Break-before-make connection helper for a named route.

		This method always calls ``open_all()`` before closing the new route so
		route changes do not momentarily short two paths together.

		Args:
			route_name:
				Route name such as ``"keithley"`` or ``"keysight"``.
			slot:
				Optional slot override for the route's channels.

		Returns:
			The normalized channels that were closed for the route.
		"""
		channels = self.get_route_channels(route_name, slot=slot)
		self.open_all()
		self.close_channel(channels)
		return channels

	def connect_keithley_smu(self, slot: int | None = None) -> list[str]:
		"""Connect the named Keithley SMU route.

		Args:
			slot:
				Optional slot override for the route's channels.

		Returns:
			The normalized channels that were closed for the route.
		"""
		return self.connect_named_route("keithley", slot=slot)

	def connect_keysight_smu(self, slot: int | None = None) -> list[str]:
		"""Connect the named Keysight SMU route.

		Args:
			slot:
				Optional slot override for the route's channels.

		Returns:
			The normalized channels that were closed for the route.
		"""
		return self.connect_named_route("keysight", slot=slot)

	def reset(self) -> None:
		"""Attempt a switch reset without raising on instrument rejection."""
		try:
			self.send_switch_command("reset()")
		except Exception:
			pass

	def close_session(self) -> None:
		"""Close the switch VISA session."""
		self._close_device()

	def close(self) -> None:
		"""Alias for ``close_session``."""
		self.close_session()

	def __del__(self) -> None:
		"""Best-effort destructor cleanup."""
		try:
			self.close_session()
		except Exception:
			pass
