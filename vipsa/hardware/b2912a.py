# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:23:20 2025

@author: amdm
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:30:54 2022

@author: darre

Library for B2912A
"""

import pyvisa
from time import sleep
import numpy as np
import logging as log

def sync(dev):
    
    """
    For synchronisation.
    To be used between other SCPI calls
    """
    
    dev.write("*OPC?")
    opc = dev.read()
    log.info(opc)
    

def send_pulse(dev,set_voltage,set_width,set_acquire_delay,read_voltage,read_width,read_acquire_delay,low,current_compliance,set_range):

    """
    Sends 4 pulses -> set, rest (0V), read, rest (0V)
    Returns the read current
    """

    dev.write("*RST")

    log.info("FORMAT")
    dev.write(":FORM:ELEM:SENS TIME,CURR")
    get_error(dev)

    # Set source

    dev.write(":SOUR1:FUNC:MODE VOLT")
    log.info("SETUP: Mode set to volt")
    get_error(dev)

    #dev.write(":SOUR1:FUNC:SHAP PULS")
    #log.info("SETUP: Pulse mode selected.")
    #get_error(dev)
    
    dev.write("SOUR1:VOLT:MODE LIST")
    log.info("SETUP: List mode activated")
    get_error(dev)


    dev.write(f":SOUR1:LIST:VOLT {set_voltage},{low},{read_voltage},{low}")
    #dev.write(f":SOUR1:LIST:VOLT 1.5,0,0.1,0")
    log.info(f"SEND_PULSE: Pulsed list sweep set to {set_voltage},{low},{read_voltage},{low}")
    get_error(dev)


    '''
    dev.write(f":SOUR1:PULS:WIDT {set_width}")
    log.info(f"SEND_PULSE: Pulse width set to {set_width}")
    get_error(dev) 
    '''
    # Set trigger

    #dev.write(":TRIG:TRAN:DEL 0")
    #log.info(f"SEND_PULSE: Transient trigger set to 0")
    #get_error(dev)   


    dev.write(":SENS1:FUNC \"CURR\"")
    log.info("SETUP: Sensing curent.")
    get_error(dev)

    # Set sens

    dev.write(":SENS1:CURR:RANG:AUTO OFF")
    log.info("SETUP: Autorange off.")
    get_error(dev)

    dev.write(f":SENS1:CURR:RANG {set_range}")
    log.info(f"SETUP: Current range set to {set_range}")
    get_error(dev)

    dev.write(f":SENS1:CURR:PROT {current_compliance}")
    log.info(f"SETUP: Current compliance set to {current_compliance}")
    get_error(dev)

    dev.write(":SENS1:CURR:APER:AUTO ON")
  

    dev.write(f":TRIG:ACQ:DEL {set_acquire_delay}")
    log.info(f"SEND_PULSE: Acquistion trigger set to {set_acquire_delay}")
    get_error(dev)     

    dev.write(":TRIG:SOUR TIM")
    dev.write(f":TRIG:TIM {set_width}") # CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dev.write(":TRIG:COUN 4")

    # Acquire!
    dev.write(":OUTP ON")
    log.info(f"Output ON, ACQUIRING")
    dev.write(":INIT (@1)")
    a= dev.query("*OPC?")


    dev.write(":FETC:ARR? (@1)")
    values = dev.read()
    a = dev.query("*OPC?")
    print(values)

    log.info(values)

    # TIME CURR STAT
    
    datadict = {'Time':[],'Current':[]}
    
    # Assume that data is all comma-separated
    
    data_split = values.split(',')
  
    
    for i in range(len(data_split)):
        
        #print(f"Current value is i = {i}")
        # current first then time, due to convention from b2912a.
        
        if np.mod(i,2) == 0:
            datadict['Current'].append(data_split[i])

        elif np.mod(i,2) == 1:
            datadict['Time'].append(data_split[i])
        else:
            # np.mod(i,3) == 2
            pass
            #datadict['Status'].append(data_split[i])
    print (datadict)        
    return float(datadict['Current'][2])

    pass

def setup_for_pulse(dev,nplc):

    pass

    # Set sens first




    # Then set source if possible. 



    pass

def list_sweep(dev,list_v,compliance,dc_nplc,pulse_nplc):

    # Set up for dc

    dev.write("*RST")
    log.info("SETUP: RESETTED")

    log.info("FORMAT")
    dev.write(":FORM:ELEM:SENS TIME,CURR,VOLT")
    get_error(dev)

    # Set sens first

    dev.write(":SENS1:FUNC \"CURR\"")
    log.info("LIST_SWEEP: Sensing curent.")
    get_error(dev)


    time_nplc = dc_nplc/50
    dev.write(f":SENS1:CURR:APER:AUTO ON")
    #log.info(f"LIST_SWEEP: NPLC set to {dc_nplc}")
    get_error(dev)


    # Then set source if possible. 


    dev.write(":SOUR1:VOLT:MODE VOLT")
    log.info("LIST_SWEEP: Mode set to volt")
    get_error(dev)

    #dev.write(":SOUR1:FUNC:SHAP DC")
    #log.info("LIST_SWEEP: DC mode selected.")
    #get_error(dev)
    
    #dev.write("SOUR1:VOLT:MODE SWE")

    dev.write(":SOUR1:VOLT:MODE LIST")
    log.info("LIST_SWEEP: Sweep mode activated")
    get_error(dev)

    # Set source

    #dev.write(":SOUR1:SWE:STA DOUB")
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    #print(dev.query(":SOUR1:SWE:STA?"))
    #dev.write(f":SOUR1:VOLT:STAR {start_volt}")
    #dev.write(f":SOUR1:VOLT:STOP {end_volt}")
    #dev.write(f":SOUR1:VOLT:POIN 122")

    stringlist = str(list_v)
    stringlist = stringlist.replace('[','')
    stringlist = stringlist.replace(']','')

    print(stringlist)

    dev.write(f":SOUR1:LIST:VOLT {stringlist}")

    dev.write(":SENS1:CURR:RANG:AUTO ON")
    log.info("SETUP: Autorange on.")
    get_error(dev)


    dev.write(f":SENS1:CURR:PROT {compliance}")
    log.info(f"SETUP: Current compliance set to {compliance}")
    get_error(dev)



    # Set trigger

    dev.write(":TRIG:TRAN:DEL 0")
    log.info(f"SEND_PULSE: Transient trigger set to 0")
    get_error(dev)

    dev.write(f":TRIG:ACQ:DEL 0.05")
    log.info(f"SEND_PULSE: Acquistion trigger set to ")
    get_error(dev)     

    dev.write(":TRIG:SOUR TIM")
    dev.write(":TRIG:TIM 0.1") # CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dev.write(f":TRIG:COUN 122")

    # Acquire!
    dev.write(":OUTP ON")
    log.info(f"Output ON, ACQUIRING")
    dev.write(":INIT (@1)")
    dev.write("*OPC?")
    a = dev.read()

    dev.write(":FETC:ARR? (@1)")
    values = dev.read()
    a = dev.query("*OPC?")
    print(values)
    log.info(values)

    setup_for_pulse(dev,pulse_nplc)

    pass




    
def set_settings_current_dlts(dev,current_range,fill_voltage,reverse_bias_voltage,measurement_interval):
    
    """
    General settings for pulsed measurement.
    """
    
    # FORMAT SETTINGS
    log.info("FORMAT")
    dev.write(":FORM:ELEM:SENS TIME,CURR")
    get_error(dev)
    
    # SENSE SETTINGS
    log.info("SENSE")
    dev.write(":SENS:FUNC:ON \"CURR\"")
    get_error(dev)
    #dev.write(":SENS:CURR:RANG:AUTO OFF")
    #get_error(dev)
    #dev.write(f":SENS:CURR:RANG:UPP {current_range}")
    #get_error(dev)
    dev.write(":SENS:CURR:PROT 0.06")
    get_error(dev)
    dev.write(":SENS:WAIT 0")
    get_error(dev)

    
    # SOURCE SETTINGS
    log.info("SOURCE")
    dev.write(":SOUR1:FUNC:MODE VOLT")
    get_error(dev)
    dev.write(":SOUR1:FUNC:SHAP DC")
    get_error(dev)
    dev.write(f":SOUR1:VOLT {fill_voltage}")
    get_error(dev)
    #dev.write(f"SOUR1:VOLT:TRIG {reverse_bias_voltage}")
    #get_error(dev)
    #dev.write(":SOUR1:PULS:DEL 0")
    #get_error(dev)
    #pulsewidth = ":SOUR1:PULSE:WIDT " + str(measurement_interval)
    #dev.write(pulsewidth)
    #get_error(dev)
    #dev.write(":PULS:DEL 0")
    #get_error(dev)
    #dev.write(":SOUR:WAIT 0")
    #get_error(dev)
    #dev.write(":SENS1:WAIT 0")
    #get_error(dev)
    
    # TRIGGER SETTINGS
    log.info("TRIGGER")
    #acq_del = measurement_interval/2
    trigacqdel = ":TRIG1:ACQ:DEL " + str(0)
    dev.write(trigacqdel)
    get_error(dev)
    dev.write(":TRIG1:TRAN:DEL 0")
    get_error(dev)
    dev.write(":TRIG1:COUN 100000")
    get_error(dev)
    dev.write(":TRIG1:SOUR TIM")
    get_error(dev)
    trigtim = ":TRIG1:TIM " + str(measurement_interval)
    dev.write(trigtim)
    get_error(dev)

    
    # TRACE SETTINGS
    log.info("TRACE")
    dev.write(":TRAC1:FEED:CONT NEV")
    get_error(dev)
    dev.write(":TRAC1:CLE")
    get_error(dev)
    dev.write(":TRAC1:POIN 100000")
    get_error(dev)
    dev.write(":TRAC1:FEED SENS")
    get_error(dev)
    dev.write(":TRAC1:FEED:CONT NEXT")
    get_error(dev)
    dev.write(":TRAC1:TST:FORM ABS")
    get_error(dev)

    
    
def get_current_range(dev,fill_voltage,initial_range):
    
    if initial_range < 1E-8:
        return 10
    else:
    
        """
        Sets range to auto, makes a measurement, then turns off auto range,
        and set range to the current setting.
        """
        
        dev.write(":SOUR1:FUNC:MODE VOLT")
        get_error(dev)
        dev.write(":SOUR1:FUNC:SHAP DC")
        get_error(dev)
        dev.write(":SENS:FUNC:ON \"CURR\"")
        get_error(dev)
        dev.write(":SENS:CURR:RANG:AUTO OFF")
        get_error(dev)
        dev.write(f":SENS:CURR:RANG:UPP {initial_range}")
        dev.write(":SENS:CURR:PROT 60e-3")
        get_error(dev)
        dev.write(":FORM:ELEM:SENS CURR")
        get_error(dev)
        sync(dev)
        
        dev.write(f":SOUR:VOLT {fill_voltage}")
        get_error(dev)
        
        # TRIGGER SETTINGS
        dev.write(":TRIG1:DEL 0")
        get_error(dev)
        dev.write(":TRIG1:COUN 1")
        get_error(dev)
        sync(dev)
        
        # TRACE SETTINGS
        dev.write(":TRAC1:FEED:CONT NEV")
        get_error(dev)
        dev.write(":TRAC1:CLE")
        get_error(dev)
        dev.write(":TRAC1:POIN 1")
        get_error(dev)
        dev.write(":TRAC1:FEED SENS")
        get_error(dev)
        dev.write(":TRAC1:FEED:CONT NEXT")
        get_error(dev)
        dev.write(":TRAC1:TST:FORM ABS")
        get_error(dev)
        sync(dev)
        
        # START MEASUREMENT
        dev.write(":OUTP1 ON")
        get_error(dev)
        dev.write(":INIT:IMM (@1)")
        get_error(dev)
        sync(dev)
        
        dev.write(":TRAC1:FEED:CONT NEV")
        get_error(dev)
        dev.write(":TRAC1:DATA?")
        curr = dev.read()
        log.info(f"Current value is {curr} A.")
        dev.write(":TRAC1:CLE")
        get_error(dev)
        
        dev.write(":OUTP1 OFF")
        get_error(dev)
        
        
        if np.abs(float(curr)) < float(initial_range/10):
            sleep(0.5)
            rangemax = get_current_range(dev, fill_voltage, initial_range/10)
            return float(rangemax)
        else:
            dev.write(":SENS:CURR:RANG:UPP?")
            rangemax = dev.read()
            log.info("Range is "+str(rangemax))
            return float(rangemax)




def start_measurement(dev,fill_duration,reverse_bias_voltage,current_range):
    
    """
    Trap-filling and transient capture process done in this function.
    """
    
    # Fill first
    initial_opc = 0
    log.info(f"STARTING. Fill for {fill_duration}s, then data capture begins.")
    log.info(f"Transient Current Range Before Start = {current_range}" )
    dev.write(":OUTP1 ON")
    sleep(fill_duration)
    dev.write(f":SOUR1:VOLT {reverse_bias_voltage}")
    dev.write(f":SENS:CURR:RANG:UPP {current_range}")
    # Start measurement
    
    dev.write(":INIT:IMM (@1)")
    
    while initial_opc == 0:
        dev.write("*OPC?")
        initial_opc = dev.read()
        log.info("OPC = " + str(initial_opc))
        
    dev.write(":OUTP1 OFF")
    

    
def get_dlts_data(dev,current_range):

    dev.write(":TRAC1:FEED:CONT NEV")
    dev.write(":TRAC1:DATA?")
    
    data = dev.read()
    sync(dev)
    log.info(data)
    # TIME CURR STAT
    
    datadict = {'Time':[],'Current':[],'CurrentRange':[]}
    
    # Assume that data is all comma-separated
    
    data_split = data.split(',')
  
    
    for i in range(len(data_split)):
        
        #print(f"Current value is i = {i}")
        # current first then time, due to convention from b2912a.
        
        if np.mod(i,2) == 0:
            datadict['Current'].append(data_split[i])
            
            # only need once, since guaranteed to pass here once.
            datadict['CurrentRange'].append(current_range) 
            
        elif np.mod(i,2) == 1:
            datadict['Time'].append(data_split[i])
        else:
            # np.mod(i,3) == 2
            pass
            #datadict['Status'].append(data_split[i])
            
        
    
    return datadict

def get_error(dev):
    
    dev.write(":SYST:ERR:ALL?")
    log.error(dev.read())
    
def get_transient_current_range(max_current, current_range):
    
    if max_current < current_range/10:
        newrange = get_transient_current_range(max_current,current_range/10)
        return newrange
    else:
        newrange = current_range
        return newrange
    
    
    
