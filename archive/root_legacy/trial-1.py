import pyvisa

ADDR = "USB0::0x05E6::0x2450::04507059::INSTR"

rm = pyvisa.ResourceManager()
inst = rm.open_resource(ADDR)

inst.write_termination = "\n"
inst.read_termination = "\n"

inst.write("*LANG SCPI")

print(inst.query(":SYST:ERR?"))