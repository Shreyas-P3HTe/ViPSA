# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 11:55:26 2025

@author: shrey
"""

from Source_Measure_Unit import KeithleySMU
from Datahandling import Data_Handler

SMU = KeithleySMU(0)
DH = Data_Handler()

ch_v = 0.001
curr = SMU.get_contact_current(ch_v)

print(f'Resistance = {ch_v/curr}')

path = "C:/Users\shrey\OneDrive\Desktop\sweep_list.csv"
save = "C:/Users/shrey/OneDrive/Desktop/0"
d, r = SMU.list_IV_sweep_split(path, pos_compliance = 1e-3, neg_compliance = 1e-2, delay=0.0001)

sweep = DH.save_file(d, 'sweep', 0, 0, curr, 0, save_directory=save)

#res = DH.save_file(r, 'resistance', 0, 0, curr, 0, save_directory=save)

DH.show_plot(sweep)

#DH.show_resistance(res, cycles=4)