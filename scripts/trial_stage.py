# -*- coding: utf-8 -*-
"""
Created on Fri May  2 16:21:11 2025

@author: shrey
"""

from vipsa.hardware.Source_Measure_Unit import KeithleySMU
from vipsa.hardware.Openflexture import stage

stage = stage('COM5', 115200, 1)
SMU = KeithleySMU(0)
SMU.smu.write(":ROUT:TERM REAR")
print("Active terminals:", SMU.smu.ask(":ROUT:TERM?").strip())
stage.move_z_by(60)
stage.disconnect()

print(SMU.get_contact_current(0.01))


