# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 17:14:05 2025

@author: amdm
"""

import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import numpy as np

csvpath = "C:/Users/amdm/OneDrive - Nanyang Technological University/ViPSA data folder/Spinbot_memristors_PEA_2/trial_PMMA/0/Resistance/device_0.csv"

df = pd.read_csv(csvpath)
res_arr = []
cycles_arr = []
cycle_no = 0
step_size = len(df) // 24
print(step_size)


for i in range(0, 12):
    start_idx = (i * 400)
    end_idx = (i + 1)*400
    res = df.iloc[start_idx:end_idx, 2] / df.iloc[start_idx:end_idx, 3]
    cycles_arr.append(cycle_no)
    
    if (i+1)%2 == 0:
        cycle_no +=1
    
    res_arr.append(abs(stat.mean(res)))
    print(abs(stat.mean(res))*1e-3,"kΩ")
    
print(cycles_arr)

for i in range (0,12):
    if i%2 == 0:
        plt.scatter(x= cycles_arr[i], y =res_arr[i], marker='.', c='red')
    else:
        plt.scatter(x= cycles_arr[i], y =res_arr[i], marker='.', c='green')

plt.xlabel("Cycle number - 0th is forming")
plt.ylabel("Resistance (kΩ, log scale)")
legend = ["On", "on", "Off", "off"]
plt.legend(legend)
plt.yscale("log")
plt.show()


