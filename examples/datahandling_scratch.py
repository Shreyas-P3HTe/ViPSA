# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 15:34:38 2025

@author: amdm
"""

from vipsa.analysis.Datahandling import Data_Handler

Handle = Data_Handler()

res_file = "C:/Users/amdm/OneDrive - Nanyang Technological University/ViPSA data folder//sample/Resistance\device_1.csv"
DCIV_file = "C:/Users/amdm/OneDrive - Nanyang Technological University/ViPSA data folder//sample/Sweep\device_1.csv"

Handle.analyze_resistance_cycles(res_file,51)
Handle.show_plot(DCIV_file)