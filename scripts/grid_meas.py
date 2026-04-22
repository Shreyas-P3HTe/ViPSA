# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:08:39 2025

@author: amdm
"""

from vipsa.analysis.Datahandling import Data_Handler
from vipsa.workflows.Main4 import Vipsa_Methods

slot_no = "Size3.1"
folder = "Sample_1" 
vipsa = Vipsa_Methods()

sweep_path = f"C:/Users/amdm/OneDrive - Nanyang Technological University/Backup/Desktop/sweep patterns/Grid/Pratap/DCIV_sweep_with_forming.csv"
save_directory = f"C:/Users/amdm/OneDrive - Nanyang Technological University/ViPSA data folder/Pratap/{folder}/{slot_no}"

grid_path = f"C:/Users/amdm/OneDrive - Nanyang Technological University/ViPSA data folder/Pratap/{folder}/{slot_no}/grid.csv"
#pulse_path = "C:/Users/amdm/Desktop/sweep patterns/Pulses_1000.csv"
vipsa.connect_equipment(SMU_name='Keithley2450')

try :
    vipsa.stage.move_z_by(-20)
      
    vipsa.measure_IV_gridwise(slot_no, grid_path, pos_compl=1e-3, neg_compl=1e-1, sweep_delay=0.002,step_size=1,
                              test_voltage= 0.1, lower_threshold= 1e-, upper_threshold= 1e-8, wait_time=0, align=False,
                              save_directory=save_directory, sweep_path=sweep_path, startpoint=1, use_4way_split=True) #<<<<< MAKE  SURE THE STARTPOINT IS CORRECT!!!

except Exception as e :
    if e == KeyboardInterrupt :
        print("Interrupted by user")
        
    else :
        print(e)

finally :        
    vipsa.disconnect_equipment()





