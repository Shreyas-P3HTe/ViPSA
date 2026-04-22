# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:08:39 2025

@author: amdm
"""

from vipsa.analysis.Datahandling import Data_Handler
from vipsa.workflows.Main4 import Vipsa_Methods

slot_no = "1"
folder = "negative reset investigation"
vipsa = Vipsa_Methods()

grid_path = f"C:/Users/amdm/OneDrive - Nanyang Technological University/ViPSA data folder/{folder}/grid.csv"
save_directory = f"C:/Users/amdm/OneDrive - Nanyang Technological University/ViPSA data folder/{folder}/"
sweep_path = "C:/Users/amdm/OneDrive - Nanyang Technological University/Backup/Desktop/sweep patterns/neg_sweep.csv"
pulse_path = "C:/Users/amdm/Desktop/sweep patterns/Pulses_1000.csv"

vipsa.connect_equipment()

try :
    vipsa.stage.move_z_by(-0)
      
    vipsa.measure_IV_gridwise(slot_no, grid_path, pos_compl=0.001, neg_compl=0.01, sweep_delay=0.01, save_directory=save_directory, sweep_path=sweep_path, startpoint=1) #redo 10 

except Exception as e :
    if e == KeyboardInterrupt :
        print("Interrupted by user")
        
    else :
        print(e)

finally :        
    vipsa.disconnect_equipment()




