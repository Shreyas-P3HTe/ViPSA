from __future__ import annotations

from typing import Dict


class FakeVisaResource:
    def __init__(
        self,
        address: str,
        responses: Dict[str, str] | None = None,
        command_log: list[str] | None = None,
        read_response: str = "0",
    ) -> None:
        self.address = address
        self.timeout = None
        self.read_termination = "\n"
        self.write_termination = "\n"
        self.command_log = command_log if command_log is not None else []
        self.responses = {"*IDN?": "FAKE,INSTRUMENT,0,0"}
        if responses:
            self.responses.update(responses)
        self.read_response = read_response

    def write(self, cmd: str) -> None:
        self.command_log.append(cmd)

    def query(self, cmd: str) -> str:
        self.command_log.append(cmd)
        return str(self.responses.get(cmd, "0"))

    def read(self) -> str:
        return str(self.read_response)

    def close(self) -> None:
        return None

    def get_command_log(self) -> list[str]:
        return list(self.command_log)


class FakeVisaRM:
    def __init__(
        self,
        responses: Dict[str, str] | None = None,
        resources: tuple[str, ...] | None = None,
    ) -> None:
        self.responses = {"*IDN?": "FAKE,INSTRUMENT,0,0"}
        if responses:
            self.responses.update(responses)
        self.resources = resources or ("USB0::FAKE::INSTR",)
        self.command_log: list[str] = []
        self._opened: dict[str, FakeVisaResource] = {}

    def list_resources(self) -> tuple[str, ...]:
        return self.resources

    def open_resource(self, address: str) -> FakeVisaResource:
        if address not in self._opened:
            self._opened[address] = FakeVisaResource(
                address=address,
                responses=self.responses,
                command_log=self.command_log,
            )
        return self._opened[address]

    def get_command_log(self) -> list[str]:
        return list(self.command_log)


def get_command_log(target) -> list[str]:
    return target.get_command_log()
