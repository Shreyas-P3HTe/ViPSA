
"""
Shreyas' code.

This module holds the instrument class for tools from Keysight.
Methods are written in SCPI. 
Make sure your environment is set to Python 3.10 and the correct drivers from keysight website are downloaded and installed.

Classes:
    KeysightSMU()
    KeithleySMU()
"""


import time
import numpy as np
import pyvisa
import pandas as pd
import logging as log
from pymeasure.instruments.keithley import Keithley2450

class KeysightSMU():
    
    def __init__(self, device_no, address=None, switch=None, switch_channel=None, connect_switch=False):
        
        rm = pyvisa.ResourceManager()
        address_list = list(rm.list_resources())
        self.address = address if address is not None else address_list[device_no]
        
        print("Devices found :" , self.address)
        
        if self.address == None : 
            print("ERROR: device could not be found")
        
        SMU = rm.open_resource(self.address) 
        print("Handshaking with : ", SMU.query("*IDN?"),"at",self.address)
        SMU.close()
        
        self.tiny_IV = "C:/Users/amdm/OneDrive - Nanyang Technological University/ViPSA data folder/sweep patterns/tinyIV.csv"
        self.resistsnce_df = pd.read_csv(self.tiny_IV)        
        # Optional switching matrix that routes SMU to a given relay channel
        self.switch = switch
        self.switch_channel = switch_channel
        self.switch_profile = "keysight"
        if connect_switch and self.switch is not None:
            try:
                self.connect_switch_path()
                print("Switch path prepared for KeysightSMU")
            except Exception as e:
                print(f"Warning: could not connect switch channel {self.switch_channel}: {e}")

    def _open_resource(self, adr=None, timeout=10000):
        if adr is None:
            adr = self.address

        rm = pyvisa.ResourceManager()
        smu = rm.open_resource(adr)
        smu.read_termination = "\n"
        smu.write_termination = "\n"
        smu.timeout = timeout
        return smu

    def connect_switch_path(self):
        if self.switch is None:
            return

        route = self.switch_channel if self.switch_channel is not None else self.switch_profile
        route_name = route.lower() if isinstance(route, str) else None

        if hasattr(self.switch, "connect_named_route") and route_name in {"keithley", "keysight"}:
            self.switch.connect_named_route(route_name)
            return

        if route_name == "all" and hasattr(self.switch, "open_all"):
            self.switch.open_all()
            return

        if hasattr(self.switch, "open_all"):
            self.switch.open_all()
        self.switch.close_channel(route)

    def disconnect_switch_path(self):
        if self.switch is None:
            return

        if hasattr(self.switch, "open_all"):
            self.switch.open_all()
            return

        if self.switch_channel is not None:
            self.switch.open_channel(self.switch_channel)

    def close_session(self):
        """KeysightSMU uses short-lived VISA sessions, so there is nothing persistent to close."""
        try:
            self.abort_measurement()
        except Exception:
            pass

        try:
            self.disconnect_switch_path()
        except Exception:
            pass
                  
    def get_address(self):
        
        #if the queried address is needed to be explicitely printed use this
        #It will also return the address so it can be assigned to a variable 
        print("The device address is", self.address)
        
        return self.address
    
    def get_error(self,SMU):
        
        SMU.write(":SYST:ERR:ALL?")
        log.error(SMU.read())
       
    def reset_device(self, adr = None):
        if adr == None:
            adr = self.address
        
        rm = pyvisa.ResourceManager()
        
        try:
            SMU = rm.open_resource(adr)
            SMU.write("*RST")
            SMU.write("*CLS")
        
        except Exception as e:
            print("Error resetting :",e)
    
    def sync(self,SMU):
        
        """
        For synchronisation.
        To be used between other SCPI calls
        """
        
        SMU.write("*OPC?")
        opc = SMU.read()
        log.info(opc)        
    
    def send_command(self, command, adr = None):
        if adr is None:
            adr = self.address
        
        rm = pyvisa.ResourceManager()
        
        try:
            SMU = rm.open_resource(adr)
            SMU.write(f"{command}")
            SMU.close()
        
        except Exception as e:
            print("Error sending :", e)

    def write(self, command, adr=None):
        smu = self._open_resource(adr=adr)
        try:
            smu.write(command)
        finally:
            smu.close()

    def ask(self, command, adr=None):
        smu = self._open_resource(adr=adr)
        try:
            return smu.query(command)
        finally:
            smu.close()

    def prepare_contact_probe(self, voltage, compliance, adr=None):
        self.connect_switch_path()
        smu = self._open_resource(adr=adr)
        try:
            smu.write("*RST")
            smu.write("*CLS")
            smu.write("SOUR:FUNC VOLT")
            smu.write('SENS:FUNC "CURR"')
            smu.write(f"SOUR:VOLT {voltage}")
            smu.write(f"SENS:CURR:PROT {compliance}")
            smu.write("OUTP ON")
        finally:
            smu.close()

    def stop_output(self, adr=None):
        smu = None
        try:
            smu = self._open_resource(adr=adr, timeout=5000)
            for command in (":OUTP OFF", "OUTP OFF", ":OUTP1 OFF", ":OUTP2 OFF"):
                try:
                    smu.write(command)
                except Exception:
                    continue
        finally:
            if smu is not None:
                smu.close()

    def abort_measurement(self, adr=None):
        smu = None
        try:
            smu = self._open_resource(adr=adr, timeout=5000)
            for command in (":ABOR", "ABOR"):
                try:
                    smu.write(command)
                except Exception:
                    continue
            for command in (":OUTP OFF", "OUTP OFF", ":OUTP1 OFF", ":OUTP2 OFF"):
                try:
                    smu.write(command)
                except Exception:
                    continue
        finally:
            if smu is not None:
                smu.close()

            
    def get_contact_current(self, voltage, compliance=10e-6, adr=None):
            
        if adr is None:
            adr = self.address
        
        rm = pyvisa.ResourceManager()
        
        try:
            SMU = rm.open_resource(adr)
            SMU.read_termination = '\n'
            SMU.write_termination = '\n'
            SMU.timeout = 10000
    
            SMU.write('*RST')
            SMU.write('*CLS')
                
            SMU.write('SOUR:FUNC VOLT')
            SMU.write('SENS:FUNC "CURR"')
            SMU.write(f'SOUR:VOLT {voltage}')
            SMU.write(f'SENS:CURR:PROT {compliance}')
                    
            SMU.write('OUTP ON')
    
            response = SMU.query('READ?').strip()
            parts = [part.strip() for part in response.split(',') if part.strip()]
            current = abs(float(parts[1] if len(parts) >= 2 else parts[0]))
            print(f": Contact_current = {current} A")
                
            SMU.write('OUTP OFF')
            SMU.close()
            return current
                
        except Exception as e:
            print("Error measuring current :", e)
            
            
    def get_contact_current_fast(self, voltage, adr=None, settle=0.02):
        """
        Lightning-fast absolute current read using bare SCPI for Keysight B2901B/B2902B.
        Assumes SMU already in voltage-source / current-measure mode.
    
        Args:
            voltage : float
                Voltage level to apply (V)
            adr : str, optional
                VISA resource address. If None, uses self.address
            settle : float
                Wait time after setting voltage before reading (s)
    
        Returns:
            float : absolute current in A
        """
        if adr is None:
            adr = self.address
    
        rm = pyvisa.ResourceManager()
    
        try:
            smu = rm.open_resource(adr)
            smu.read_termination = '\n'
            smu.write_termination = '\n'
            smu.timeout = 5000
    
            # Assume SMU already configured as voltage source & current measure.
            # Just set voltage and read current.
            smu.write(f":SOUR:VOLT {voltage}")
            smu.write(":OUTP ON")
    
            if settle > 0:
                time.sleep(settle)
    
            # "READ?" performs a measure and returns the value(s)
            # Keysight returns comma-separated values: V, I, optional timestamp, etc.
            resp = smu.query("READ?")
            parts = resp.strip().split(',')
    
            # Extract current (second value if available)
            if len(parts) >= 2:
                current = abs(float(parts[1]))
            else:
                # Sometimes only one value is returned if source only
                current = abs(float(parts[0]))
    
            return current
    
        except Exception as e:
            print("Error measuring fast current:", e)
            return 0.0
    
        finally:
            try:
                smu.write(":OUTP OFF")
                smu.close()
            except Exception:
                pass

    
    def split_list(self, vlist):
    
        """
        Takes in the volatge list and splits it into positive and negative cylces.
        
        Arguments : 
            vlist : list of voltages as a list of strings/ float
        
        Returns : 
            voltage_data : List of Lists with split data containing [[cycle no, "p" or "n" tag, [cycle_vlist]]]
                    
        """
          # Ensure vlist is a numpy array of floats
        voltage_data = []
        cycle_no = 0
        current_cycle = []
        current_tag = ""
        
        for i in range(len(vlist)):
            v = vlist[i]
        
            if i == 0:  # Initialize the first cycle
                if v >= 0:
                    current_tag = "p"
                    
                else:
                    current_tag = "n"
                current_cycle.append(v)
                continue
    
    
            if v >= 0 and current_tag == "n":  # Transition from negative to positive
                cycle_no += 1
                voltage_data.append([cycle_no, current_tag, current_cycle])
                current_tag = "p"
                current_cycle = [v]
        
            elif v < 0 and current_tag == "p":  # Transition from positive to negative
                cycle_no += 1
                voltage_data.append([cycle_no, current_tag, current_cycle])
                current_tag = "n"
                current_cycle = [v]
        
            else:  # Continue current cycle
                current_cycle.append(v)
    
        # Append the last cycle
        if current_cycle:
            cycle_no += 1 # Important: Increment cycle number for the last cycle.
            voltage_data.append([cycle_no, current_tag, current_cycle])
        
        return voltage_data                        
        
    def split_list_by_4(self, vlist):
        """
        Takes in the voltage list and splits it into 4 parts:
        - pf: positive forward (0 to max)
        - pb: positive backward (max to 0)
        - nf: negative forward (0 to min)
        - nb: negative backward (min to 0)
        
        Arguments: 
            vlist : list of voltages as a list of strings/float
            
        Returns: 
            voltage_data : List of Lists with split data containing [[cycle no, cycle tag, [cycle_vlist]]]
        """
        # Ensure vlist is a numpy array of floats
        voltage_data = []
        cycle_no = 0
        current_cycle = []
        current_tag = ""
        
        pmax = max(vlist)
        nmax = min(vlist)
        
        for i in range(len(vlist)):
            v = vlist[i]
            
            if i == 0:  # Initialize the first cycle
                if v > 0: 
                    current_tag = "pf"
                elif v < 0:
                    current_tag = "nf"
                current_cycle.append(v)
                continue
            
            # Transition logic for pf (positive-forward)
            if v > 0 and current_tag == "nf":  # Transition from nf to pf
                cycle_no += 1
                voltage_data.append([cycle_no, current_tag, current_cycle])
                current_tag = "pf"
                current_cycle = [v]
            
            # Transition logic for pb (positive-backward)
            elif v < pmax and v > 0 and current_tag == "pf":  # Transition from pf to pb
                cycle_no += 1
                voltage_data.append([cycle_no, current_tag, current_cycle])
                current_tag = "pb"
                current_cycle = [v]
            
            # Transition logic for nf (negative-forward)
            elif v < 0 and current_tag == "pf":  # Transition from pf to nf
                cycle_no += 1
                voltage_data.append([cycle_no, current_tag, current_cycle])
                current_tag = "nf"
                current_cycle = [v]
            
            # Transition logic for nb (negative-backward)
            elif v > nmax and v < 0 and current_tag == "nf":  # Transition from nf to nb
                cycle_no += 1
                voltage_data.append([cycle_no, current_tag, current_cycle])
                current_tag = "nb"
                current_cycle = [v]
    
            # Continue with the current cycle
            current_cycle.append(v)
    
        # Append the last cycle
        if current_cycle:
            cycle_no += 1  # Increment cycle number for the last cycle
            voltage_data.append([cycle_no, current_tag, current_cycle])
        
        return voltage_data
        
    def simple_IV_sweep(self, vstart, vstop, vstep, compliance, delay, adr = None):
            
        #simple quick IV sweeping function. Address is assigned to self.address if 
        # by default. User can change this by passing it as an argument
            
        if adr is None:
            adr = self.address
        
        rm = pyvisa.ResourceManager()
        self.connect_switch_path()
        
        try:
            SMU = rm.open_resource(adr)
            SMU.write("*RST")
            SMU.write("*CLS")
            SMU.write("SOUR:FUNC VOLT")
            SMU.write(f"SOUR:VOLT:START {vstart}")
            SMU.write(f"SOUR:VOLT:STOP {vstop}")
            SMU.write(f"SOUR:VOLT:STEP {vstep}")
            SMU.write(f"SENS:FUNC \"CURR\"")
            SMU.write(f"SENS:CURR:PROT {compliance}")
            SMU.write("FORM:ELEM CURR")
            SMU.write("OUTP ON")
            SMU.write("SOUR:VOLT:MODE SWE")
            SMU.write(f"TRIG:DEL {delay}")
            SMU.write("TRAC:POIN 1000")
            SMU.write("TRAC:CLE")
            SMU.write("TRAC:FEED SENS")
            SMU.write("TRAC:FEED:CONT NEXT")
            SMU.write("INIT")
            time.sleep((vstop-vstart)/vstep*delay)
            SMU.query("*OPC?")
            data = SMU.query("TRAC:DATA?")
            data = list(map(float, data.split(',')))
            SMU.write("OUTP OFF")
            SMU.close()
            return data
        except Exception as e:
            print("Error measuring current :", e)
        finally:
            self.disconnect_switch_path()
     
        
    def list_IV_sweep_manual(self, csv_path, pos_compliance, neg_compliance, delay=None, adr=None):
                
        # General purpose list scanning function. 
        # The CSV file for voltage lists can be generated using the listmaker app.
        # By default, the listmaker generated timing is used from CSV.
        # Delay time can be overwritten simply by passing it as an argument, the minimum delay is 50ms
            
            
        if adr is None:
            adr = self.address
        
        rm = pyvisa.ResourceManager()
        self.connect_switch_path()
        
        try :
            SMU = rm.open_resource(adr)
            
            df = pd.read_csv(csv_path)
            print(df)
            vlist = df.iloc[:, 1].tolist()  
            tlist = df.iloc[:, 0].tolist()
            
            SMU.write('*RST')
            SMU.write('*CLS')
            
            SMU.write('SOUR:FUNC VOLT')
            SMU.write('SENS:FUNC "CURR"')
            SMU.write('FORM:ELEM CURR')
            SMU.write('OUTP ON')
            
            data = []
            column = 0
    
            if delay is None:
                t0 = time.perf_counter()
    
            for voltage in vlist:
                
                if voltage >= 0:
                    SMU.write(f'SENS:CURR:PROT {pos_compliance}')
                else:
                    SMU.write(f'SENS:CURR:PROT {neg_compliance}')
    
                SMU.write(f'SOUR:VOLT {voltage}')
                SMU.query('*OPC?')
                current = float(SMU.query('READ?'))
                
                if delay is None and column < len(tlist):
                    target = float(tlist[column])
                    now = time.perf_counter()
                    sleep_for = (t0 + target) - now
                    if sleep_for > 0.0:
                        time.sleep(sleep_for)
                    timestamp = target
                    column += 1
                    
                elif delay is not None and delay <= 0.05:
                    delay = 0.05
                    timestamp = delay * column
                    column += 1
                    time.sleep(delay)
                    
                else :
                    timestamp = delay * column
                    column += 1
                    time.sleep(delay)
                
                data.append((timestamp, float(voltage), current))
    
            SMU.write('OUTP OFF')
            SMU.close()
            
            data = np.array(data)
            
            return data
        
        except Exception as e:
            print("ERROR during list sweep:", e)
        finally:
            self.disconnect_switch_path()


    def scan_read_vlist(self, dev, voltage_list, set_width, set_acquire_delay, current_compliance, set_range=None):
    
        """
        Sends the list of voltages
        Returns the read current
        """
        try :
            print("I am in scan_read_vlist")
            
            voltage_string = str(voltage_list)
            voltage_string = voltage_string.replace('[','')
            voltage_string = voltage_string.replace(']','')
            pulse_no = len(voltage_list)
            print(pulse_no)
            
            dev.write("*RST")
        
            log.info("FORMAT")
            dev.write(":FORM:ELEM:SENS TIME,CURR")
            #get_error(dev)
        
            # Set source
        
            dev.write(":SOUR1:FUNC:MODE VOLT")
            log.info("SETUP: Mode set to volt")
            #get_error(dev)
        
            # dev.write(":SOUR1:FUNC:SHAP PULS")
            # log.info("SETUP: Pulse mode selected.")
            #get_error(dev)
            
            dev.write("SOUR1:VOLT:MODE LIST")
            log.info("SETUP: List mode activated")
            #get_error(dev)
        
        
            dev.write(f":SOUR1:LIST:VOLT {voltage_string}")
            #dev.write(f":SOUR1:LIST:VOLT 1.5,0,0.1,0")
            log.info(f"SEND_PULSE: Pulsed list sweep set to {voltage_string}")
            #get_error(dev)
        
        
            # dev.write(f":SOUR1:PULS:WIDT {set_width}")
            # log.info(f"SEND_PULSE: Pulse width set to {set_width}")
            # #get_error(dev) 
            
            # Set trigger
        
            dev.write(":TRIG:TRAN:DEL 0")
            log.info(f"SEND_PULSE: Transient trigger set to 0")
            #get_error(dev)   
        
        
            dev.write(":SENS1:FUNC \"CURR\"")
            log.info("SETUP: Sensing curent.")
            #get_error(dev)
        
            # Set sens
        
            dev.write(":SENS1:CURR:RANG:AUTO OFF")
            log.info("SETUP: Autorange off.")
            #get_error(dev)
        
            if set_range != None :
                dev.write(f":SENS1:CURR:RANG {set_range}")
                log.info(f"SETUP: Current range set to {set_range}")
                #get_error(dev)
                
            else :
                dev.write(":SENS1:CURR:RANG AUTO")
                log.info("SETUP: Current range set to AUTO")
        
            dev.write(f":SENS1:CURR:PROT {current_compliance}")
            log.info(f"SETUP: Current compliance set to {current_compliance}")
            #get_error(dev)
        
            dev.write(":SENS1:CURR:APER:AUTO ON")
          
        
            dev.write(f":TRIG:ACQ:DEL {set_acquire_delay}")
            log.info(f"SEND_PULSE: Acquistion trigger set to {set_acquire_delay}")
            #get_error(dev)     
        
            dev.write(":TRIG:SOUR TIM")
            dev.write(f":TRIG:TIM {set_width}") # CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            dev.write(f":TRIG:COUN {pulse_no}")
        
            # Acquire!
            dev.write(":OUTP ON")
            log.info(f"Output ON, ACQUIRING")
            dev.write(":INIT (@1)")
            a= dev.query("*OPC?")
        
        
            dev.write(":FETC:ARR? (@1)")
            values = dev.read()
            a = dev.query("*OPC?")
            print("values obtained:")
            #print(values)
        
            log.info(values)
            
        except Exception as e :
            print("Error in the scan_read_vlist:", e)
        
        # TIME CURR STAT
        
        currents = []
        timestamps = []
        datadict = []
        
        # Assume that data is all comma-separated
        
        data_split = values.split(',')
      
        
        for i in range(len(data_split)):
            
            #print(f"Current value is i = {i}")
            # current first then time, due to convention from b2912a.
            
            if np.mod(i,2) == 0:
                currents.append(data_split[i])
    
            elif np.mod(i,2) == 1:
                timestamps.append(data_split[i])
            else:
                # np.mod(i,3) == 2
                pass
                #datadict['Status'].append(data_split[i])
        #print (timestamps, currents)
        
        for i in range(len(voltage_list)):
            
            datadict.append((timestamps[i], voltage_list[i], currents[i]))
        
        #print(datadict)
        
        datadict = np.array(datadict)
            
        return datadict
       
    def list_IV_sweep_split(self, csv_path, pos_compliance, neg_compliance, SMU_range = None, 
                            delay = None, acq_delay = None, adr = None, pos_channel = None, neg_channel = None):     
        
        """
        Arguments
        ----------
            csv_path : String
                Sweep file.
            pos_compliance : float, Amps
                Compliance for the positive sweep.
            neg_compliance : float, Amps
                Compliance for the negative sweep.
            delay : float, optional, Seconds
                Delay between successive measurements. The default is None.
            acq_delay : float, optional, Seconds
                Where within the voltage step should the measurement be taken ?
                If None, default value is set to 1/2 delay
            adr : int, optional
                Equipemnt adress. The default is None.
            pos_channel : TYPE, optional
                Channel for the positive sweep. The default is None.
            neg_channel : TYPE, optional
                Channel for the negative sweep. The default is None.
                This is useful for

        Returns
        -------
        data as an array
        """

        if adr is None:
            adr = self.address
            print("adr")

        rm = pyvisa.ResourceManager()

        file_df = pd.read_csv(csv_path, dtype=float) #read csv without header
        

        if delay is None:
            delay = file_df.iloc[2, 0] - file_df.iloc[1, 0] #access time via iloc

        if acq_delay is None:
            acq_delay = delay / 2
        
        if SMU_range is None :
            SMU_range = 10e-3
        
        vlist = file_df.iloc[:,1].tolist()
        splits = self.split_list(vlist)
        vlist_resistance = self.resistsnce_df.iloc[:,1].tolist()
        self.connect_switch_path()

        try:

            rm = pyvisa.ResourceManager()
            SMU = rm.open_resource(adr)
            SMU.timeout = 30000000
            data_array = []
            resistance_array = []

            for split in splits:

                compliance = (pos_compliance if split[1] == 'p' else neg_compliance)
                SMU_range = (10e-2 if split[1]=='p' else 10e-2)
                #channel = (pos_channel if split[1] == 'p' else neg_channel)  # channel swapping not enabled yet
                print("Cycle n0.:",split[0], " compliance :", compliance)
                vlist_cycle = split[2]
                #print(vlist_cycle)
                #data = self.general_channel_pulsing(SMU, measurement_type='single', mode=1, positive_voltages=vlist_cycle, set_width=delay, 
                #                                    set_acquire_delay=acq_delay, current_compliance=compliance, set_range=SMU_range)
                                    
                if split[1] == 'p' : #The device is pristine or in HRS
                    res_range = 10e-8
                    res_compliance = 10e-6
                
                elif split[1] == 'n' : #The device is in LRS
                    res_range = 10e-4
                    res_compliance = 10e-3
                
                data_resistance =  self.scan_read_vlist(SMU, voltage_list = vlist_resistance, 
                                                        set_width=10e-4, set_acquire_delay=5e-4,
                                                        current_compliance = res_compliance,
                                                        set_range= res_range)
                    
                
                data = self.scan_read_vlist(SMU, voltage_list = vlist_cycle, set_width = delay, set_acquire_delay = acq_delay, 
                                             current_compliance = compliance, set_range = SMU_range)
                
                
                data_array.extend(data)
                resistance_array.extend(data_resistance)
                #print(vlist_cycle,"\n", compliance, channel, delay, acq_delay)

            return data_array, resistance_array

        except Exception as e:
            print("ERROR during list sweep:", e)
        finally:
            self.disconnect_switch_path()

    def list_IV_sweep_split_4(self, csv_path, pos_compliance, neg_compliance, SMU_range = None, 
                            delay = None, acq_delay = None, adr = None, pos_channel = None, neg_channel = None):     
        
        """
        Arguments
        ----------
            csv_path : String
                Sweep file.
            pos_compliance : float, Amps
                Compliance for the positive sweep.
            neg_compliance : float, Amps
                Compliance for the negative sweep.
            delay : float, optional, Seconds
                Delay between successive measurements. The default is None.
            acq_delay : float, optional, Seconds
                Where within the voltage step should the measurement be taken ?
                If None, default value is set to 1/2 delay
            adr : int, optional
                Equipemnt adress. The default is None.
            pos_channel : TYPE, optional
                Channel for the positive sweep. The default is None.
            neg_channel : TYPE, optional
                Channel for the negative sweep. The default is None.
                This is useful for

        Returns
        -------
        data as an array
        """

        if adr is None:
            adr = self.address
            print("adr")

        rm = pyvisa.ResourceManager()

        file_df = pd.read_csv(csv_path, dtype=float) #read csv without header
        

        if delay is None:
            delay = file_df.iloc[2, 0] - file_df.iloc[1, 0] #access time via iloc

        if acq_delay is None:
            acq_delay = delay / 2
        
        if SMU_range is None :
            SMU_range = 10e-3
        
        vlist = file_df.iloc[:,1].tolist()
        splits = self.split_list_by_4(vlist) #pf, pb, nf, nb
        vlist_resistance = self.resistsnce_df.iloc[:,1].tolist()
        self.connect_switch_path()

        try:

            rm = pyvisa.ResourceManager()
            SMU = rm.open_resource(adr)
            SMU.timeout = 30000000
            data_array = []
            resistance_array = []

            for split in splits:

                compliance = (pos_compliance if (split[1] == 'pf' or split[1] == 'pb') else neg_compliance)
                SMU_range = (10e-3 if split[1]=='pf' else 10e-4)
                #channel = (pos_channel if split[1] == 'p' else neg_channel)  # channel swapping not enabled yet
                print("Cycle n0.:",split[0], " compliance :", compliance)
                vlist_cycle = split[2]
                #print(vlist_cycle)
                #data = self.general_channel_pulsing(SMU, measurement_type='single', mode=1, positive_voltages=vlist_cycle, set_width=delay, 
                #                                    set_acquire_delay=acq_delay, current_compliance=compliance, set_range=SMU_range)
                                    
                if (split[1] == 'nf' or split[1] == 'nb') : #The device is SET
                    res_range = 10e-8
                    res_compliance = 10e-6
                
                elif (split[1] == 'pf' or split[1] == 'pb') : #The device is in RESET
                    res_range = 10e-4
                    res_compliance = 10e-3
                
                if (split[1] == 'pb' or split[1] == 'nb'):
                    data_resistance =  self.scan_read_vlist(SMU, voltage_list = vlist_resistance, 
                                                            set_width=10e-4, set_acquire_delay=5e-4,
                                                            current_compliance = res_compliance,
                                                            set_range= res_range)
                    
                
                data = self.scan_read_vlist(SMU, voltage_list = vlist_cycle, set_width = delay, set_acquire_delay = acq_delay, 
                                             current_compliance = compliance, set_range = SMU_range)
                
                
                data_array.extend(data)
                resistance_array.extend(data_resistance)
                #print(vlist_cycle,"\n", compliance, channel, delay, acq_delay)

            return data_array, resistance_array

        except Exception as e:
            print("ERROR during list sweep:", e)
        finally:
            self.disconnect_switch_path()

    
    def pulsed_measurement(self, csv_path, current_compliance, set_width=0.01, bare_list = None, set_acquire_delay = None, adr = None):
        
        if adr == None:
            adr = self.address
            
        if set_acquire_delay == None :
            set_acquire_delay = set_width/2
        self.connect_switch_path()
        
        try :
            
            rm = pyvisa.ResourceManager()
            SMU = rm.open_resource(adr)
            print("SMU is : ", SMU)
            SMU.timeout = 30000000
            
            
            if bare_list :
                
                vlist = bare_list
                
            else:
                df = pd.read_csv(csv_path)
                print(df)
                vlist = df.iloc[:, 1].tolist()  
                tlist = df.iloc[:, 0].tolist()
            
            print("now calling scan read bla bla")
            data = self.scan_read_vlist(SMU, vlist, set_width, set_acquire_delay, current_compliance, 10e-3)
            #print (data)
            return data
            
            SMU.close()
        except Exception as e:
            print("ERROR during list sweep:", e) 
        finally:
            self.disconnect_switch_path()
            
    def response_dealer(self,raw_response):

    # Data is given as source, current, time
    # Split by commas and go by mod3

        results = {'Source':[],'Current':[],'Time':[]}

        split_arr = raw_response.split(',')

        for i in range(len(split_arr)):

            if np.mod(i,3) == 0:
                 
                # Means source
                results["Source"].append(float(split_arr[i]))

            elif np.mod(i,3) == 1:

                # Means current
                results["Current"].append(float(split_arr[i]))
            
            else:

                # Means mod gives 2, so is time

                results["Time"].append(float(split_arr[i]))


        return results    
    
    def split_pulse_for_2_chan (self, vlist):
        vlist_p = []
        vlist_n = []
        for voltage in vlist :
            
            if voltage >0 :
                vlist_p.append(voltage)
                vlist_n.append(0)
            
            if voltage<0:
                vlist_p.append(0)
                vlist_n.append(-1*voltage)
                
        return vlist_p, vlist_n
            
    def general_channel_pulsing(self, adr = None, measurement_type='single', mode=1, positive_voltages=None, negative_voltages=None, set_width=50e-3, set_acquire_delay=None, current_compliance=1e-5, set_range=1e-5):

        """
        
        2025-03-10 1st Version
        
        General function for PULSED list measurements. 
        Note that this leaves the outputs ON in preparation for repeat runs, so remember to turn off the output after the entire scan is done. 

        measurement_type:   'single' or 'double'. 'single' means running one channel only. 'double' makes use of both channels, where both channels are activated together. H
                            Here the purpose is to apply an overall voltage to the DUT using two channels. Default value 'single'.

        mode:               1 or 2. In 'single' mode, refers to the channel. In 'double mode', mode=1 means that channel 1 will source the positive voltages, and current read is
                            from channel 1. Channel 2 supplies the negative voltages. In mode=2 means that channel 2 will now source the positive voltages, and current read is also
                            from channel 2, while channel 1 supplies the negative voltages. Default value 1.

        positive_voltages:  A list of positive voltages for the pulsed list measurements. Default value None. In VOLTS.
        
        negatve_voltages:   A list of negative voltages for the pulsed list measurements. In 'single' mode use None. Default value None. In VOLTS.

        set_width:          Width of pulse. From documentation the pulse is defined as when it rises to 10% of the peak value, and the width ends when it falls to 90% of the peak value.
                            This value MUST BE LARGER than the set_acquire_delay. Default value 50e-3. In SECONDS.

        set_acquire_delay:  The time in the pulse where the current is read. This MUST BE SMALLER than the set_width. If None, set_acquire_delay will be assumed to be 60% of the set_width
                            Default value None. In SECONDS.
        
        current_compliance: Compliance for current. In 'double' mode, controls both channels. Default value 1e-5. In AMPERES.

        set_range:          Sets the current range. Autorange is turned off. In 'double' mode, controls both channels. Default value 1e-5. In AMPERES.

        
        Returns three lists: A list of timestamps, a list of currents, a list of sourced voltages.
        Returns three Nones if there is an error. Check the log for output. 
        """

        #################################################### VARIABLE CHECKS #####################################################

        # print ("SMU: ", SMU, "\n Measurement_type : ", measurement_type, "\n mode : ", mode, "\n Voltages : ",  )

        # if set_acquire_delay == None:
        #     set_acquire_delay = 0.6*set_width

        # if SMU == None:
        #     log.error("GENERAL_CHANNEL_PULSING: No SMU object found. Returning 3 Nones.")
        #     return None,None,None
        
        # if measurement_type != 'single' or measurement_type != 'double':
        #     log.error(f"GENERAL_CHANNEL_PULSING: Invalid measurement type: {measurement_type}. Only accepts 'single' or 'double'. Returning 3 Nones.")
        #     return None,None,None
        
        # if mode != 1 or mode !=2:
        #     log.error(f"GENERAL_CHANNEL_PULSING: Invalid mode: {mode}. Only accepts 1 or 2. Returning 3 Nones.")
        #     return None,None,None

        # if positive_voltages == None:
        #     log.error("GENERAL_CHANNEL_PULSING: No voltage list provided. Returning 3 Nones.")
        #     return None,None,None
        


        ####################################################    MEASUREMENTS    ###################################################
        
        if adr == None:
            
            adr = self.address
            
        
        rm = pyvisa.ResourceManager()
        SMU = rm.open_resource(adr)
        print("SMU is : ", SMU)
        SMU.timeout = 30000000
            
        
        if measurement_type == 'single':

            ### Single channel pulsing. Mode is then the channel number.

            if mode == 1:
                channel = 1
                #other_channel = 2
            else:
                channel = 2
                #other_channel = 1

            SMU.write(f":TRAC{channel}:CLE")  # Clear the buffer
            SMU.write(f":TRAC{channel}:POIN {len(positive_voltages)}")  # Set the number of points to store in the buffer
        
            
            SMU.write(":FORM:ELEM:SENS SOUR,CURR,TIME")
            
            SMU.write(f":TRAC{channel}:FEED SENS")  # Select the source of data to be stored in the buffer
            SMU.write(f":TRAC{channel}:TST:FORM ABS")
            log.info("GENERAL_CHANNEL_PULSING: Timestamp Format set to ABSOLUTE.")
            SMU.write(f":TRAC{channel}:FEED:CONT NEXT")  # Enable buffer to store readings

            ################# ACTIVATE SENS FUNCTION???

            SMU.write(f":SENS{channel}:CURR:RANG:AUTO OFF")
            log.info(f"GENERAL_CHANNEL_PULSING: Channel {channel} Autorange off.")
            #SMU.write(f":SENS{other_channel}:CURR:RANG:AUTO OFF")
            #log.info(f"SETUP: Channel {other_channel} Autorange off.")

            SMU.write(f":SENS{channel}:CURR:RANG {set_range}")
            log.info(f"GENERAL_CHANNEL_PULSING: Channel {channel} Current range set to {set_range}")
            #SMU.write(f":SENS{other_channel}:CURR:RANG {set_range}")
            #log.info(f"SETUP: Channel {other_channel} Current range set to {set_range}")

            SMU.write(f":SENS{channel}:CURR:PROT {current_compliance}")
            log.info(f"GENERAL_CHANNEL_PULSING: Channel {channel} Current compliance set to {current_compliance}")
            #SMU.write(f":SENS{other_channel}:CURR:PROT {current_compliance}")
            #log.info(f"SETUP: Channel {other_channel} Current compliance set to {current_compliance}")
            
            # Set source

            # Voltage is in list
            # conert list to string and remove [ ]

            voltage_list = positive_voltages
            voltage_string = str(voltage_list)
            voltage_string = voltage_string.replace('[','')
            voltage_string = voltage_string.replace(']','')

            #neg_voltage_list = negative_voltages
            #voltage_string_2 = str(neg_voltage_list)
            #voltage_string_2 = voltage_string_2.replace('[','')
            #voltage_string_2 = voltage_string_2.replace(']','')

            SMU.write(f":SOUR{channel}:FUNC:MODE VOLT")
            #SMU.write(f":SOUR{other_channel}:FUNC:MODE VOLT")

            SMU.write(f":SOUR{channel}:FUNC:SHAP PULS") ########################################## 2025-03-07 REMOVED DUE TO USELESSNESS
            #SMU.write(f":SOUR{other_channel}:FUNC:SHAP PULS")

            SMU.write(f":SOUR{channel}:VOLT:MODE LIST")
            #SMU.write(f":SOUR{other_channel}:VOLT:MODE LIST")
            SMU.write(f":SOUR{channel}:LIST:VOLT {voltage_string}")
            log.info(f"GENERAL_CHANNEL_PULSING: Channel {channel} Pulsed list sweep set to {voltage_string}")
            #SMU.write(f":SOUR{other_channel}:LIST:VOLT {voltage_string_2}")
            #log.info(f"SEND_PULSE: Channel {other_channel} Pulsed list sweep set to {voltage_string_2}")

            SMU.write(f":SOUR{channel}:PULS:WIDT {set_width}")
            SMU.write(f":SOUR{channel}:PULS:DEL 0")
            log.info(f"GENERAL_CHANNEL_PULSING: Channel {channel} Pulse width set to {set_width}")
            #SMU.write(f":SOUR{other_channel}:PULS:WIDT {set_width}")
            #log.info(f"SEND_PULSE: Channel {other_channel} Pulse width set to {set_width}")

            # Set trigger

            SMU.write(f":TRIG{channel}:TRAN:DEL 0")
            #SMU.write(f":TRIG{other_channel}:TRAN:DEL 0")
            log.info(f"GENERAL_CHANNEL_PULSING: Transient trigger set to 0")  

            SMU.write(f":TRIG{channel}:ACQ:DEL {set_acquire_delay}")
            #SMU.write(f":TRIG{other_channel}:ACQ:DEL {set_acquire_delay}")
            log.info(f"GENERAL_CHANNEL_PULSING: Acquistion trigger set to {set_acquire_delay}")

            #SMU.write(f":TRIG{channel}:SOUR TIM")
            SMU.write(f":TRIG{channel}:TIM {set_width}") # CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #SMU.write(f":TRIG{other_channel}:SOUR TIM")
            #SMU.write(f":TRIG{other_channel}:TIM {set_width}") # CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            SMU.write(f":TRIG{channel}:COUN {len(positive_voltages)}")
            #SMU.write(f":TRIG{other_channel}:COUN {len(positive_voltages)}")
            # Acquire!
            
            #SMU.write(":SYST:GRO (@1,2)")
            SMU.write(f":OUTP{channel} ON")
            #SMU.write(f":OUTP{other_channel} ON")        
            log.info(f"GENERAL_CHANNEL_PULSING: BOTH Outputs ON, ACQUIRING from channel {mode}")
            SMU.write(f":INIT (@1)")

            # To stop code while acquiring
            self.sync(SMU)
            log.info(f"GENERAL_CHANNEL_PULSING: Data acquired. Fetching...")
            
            SMU.write(f":TRAC{channel}:DATA?")
            #SMU.write(f":FETC:ARR? (@{channel})")
            values = SMU.read()
            #SMU.write(":SYST:GRO (@1),(@2)")
            log.info(values)
            
            #print(values)

        else:

            ### Double channel pulsing. Mode is which channel will be assigned positive channel, and which channel will measurements be done from.
        
            if negative_voltages == None:
                log.error(f"GENERAL_CHANNEL_PULSING: No negative voltage list provided even though measurement_type == {measurement_type}. Returning 3 Nones.")
                return None,None,None

            if mode == 1:
                channel = 1
                other_channel = 2
            else:
                channel = 2
                other_channel = 1

            SMU.write(f":TRAC{channel}:CLE")  # Clear the buffer
            SMU.write(f":TRAC{channel}:POIN {len(positive_voltages)}")  # Set the number of points to store in the buffer
        
            
            SMU.write(":FORM:ELEM:SENS SOUR,CURR,TIME")
            
            SMU.write(f":TRAC{channel}:FEED SENS")  # Select the source of data to be stored in the buffer
            SMU.write(f":TRAC{channel}:TST:FORM ABS")
            log.info("GENERAL_CHANNEL_PULSING: Timestamp Format set to ABSOLUTE.")
            SMU.write(f":TRAC{channel}:FEED:CONT NEXT")  # Enable buffer to store readings

            ################# ACTIVATE SENS FUNCTION???

            SMU.write(f":SENS{channel}:CURR:RANG:AUTO OFF")
            log.info(f"GENERAL_CHANNEL_PULSING: Channel {channel} Autorange off.")
            SMU.write(f":SENS{other_channel}:CURR:RANG:AUTO OFF")
            log.info(f"GENERAL_CHANNEL_PULSING: Channel {other_channel} Autorange off.")

            SMU.write(f":SENS{channel}:CURR:RANG {set_range}")
            log.info(f"GENERAL_CHANNEL_PULSING: Channel {channel} Current range set to {set_range}")
            SMU.write(f":SENS{other_channel}:CURR:RANG {set_range}")
            log.info(f"GENERAL_CHANNEL_PULSING: Channel {other_channel} Current range set to {set_range}")

            SMU.write(f":SENS{channel}:CURR:PROT {current_compliance}")
            log.info(f"GENERAL_CHANNEL_PULSING: Channel {channel} Current compliance set to {current_compliance}")
            SMU.write(f":SENS{other_channel}:CURR:PROT {current_compliance}")
            log.info(f"GENERAL_CHANNEL_PULSING: Channel {other_channel} Current compliance set to {current_compliance}")
            
            # Set source

            # Voltage is in list
            # conert list to string and remove [ ]

            voltage_list = positive_voltages
            voltage_string = str(voltage_list)
            voltage_string = voltage_string.replace('[','')
            voltage_string = voltage_string.replace(']','')

            neg_voltage_list = negative_voltages
            voltage_string_2 = str(neg_voltage_list)
            voltage_string_2 = voltage_string_2.replace('[','')
            voltage_string_2 = voltage_string_2.replace(']','')

            SMU.write(f":SOUR{channel}:FUNC:MODE VOLT")
            SMU.write(f":SOUR{other_channel}:FUNC:MODE VOLT")

            SMU.write(f":SOUR{channel}:FUNC:SHAP PULS") 
            SMU.write(f":SOUR{other_channel}:FUNC:SHAP PULS")

            SMU.write(f":SOUR{channel}:VOLT:MODE LIST")
            SMU.write(f":SOUR{other_channel}:VOLT:MODE LIST")
            SMU.write(f":SOUR{channel}:LIST:VOLT {voltage_string}")
            log.info(f"GENERAL_CHANNEL_PULSING: Channel {channel} Pulsed list sweep set to {voltage_string}")
            SMU.write(f":SOUR{other_channel}:LIST:VOLT {voltage_string_2}")
            log.info(f"GENERAL_CHANNEL_PULSING:  Channel {other_channel} Pulsed list sweep set to {voltage_string_2}")

            SMU.write(f":SOUR{channel}:PULS:WIDT {set_width}")
            SMU.write(f":SOUR{channel}:PULS:DEL 0")
            log.info(f"GENERAL_CHANNEL_PULSING: Channel {channel} Pulse width set to {set_width}")
            SMU.write(f":SOUR{other_channel}:PULS:WIDT {set_width}")
            SMU.write(f":SOUR{other_channel}:PULS:DEL 0")
            log.info(f"GENERAL_CHANNEL_PULSING: Channel {other_channel} Pulse width set to {set_width}")

            # Set trigger

            SMU.write(f":TRIG{channel}:TRAN:DEL 0")
            SMU.write(f":TRIG{other_channel}:TRAN:DEL 0")
            log.info(f"GENERAL_CHANNEL_PULSING: Transient trigger set to 0")  

            SMU.write(f":TRIG{channel}:ACQ:DEL {set_acquire_delay}")
            SMU.write(f":TRIG{other_channel}:ACQ:DEL {set_acquire_delay}")
            log.info(f"GENERAL_CHANNEL_PULSING: Acquistion trigger set to {set_acquire_delay}")

            #SMU.write(f":TRIG{channel}:SOUR TIM")
            SMU.write(f":TRIG{channel}:TIM {set_width}") # CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #SMU.write(f":TRIG{other_channel}:SOUR TIM")
            SMU.write(f":TRIG{other_channel}:TIM {set_width}") # CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            SMU.write(f":TRIG{channel}:COUN {len(positive_voltages)}")
            SMU.write(f":TRIG{other_channel}:COUN {len(positive_voltages)}")

            # Acquire!
            
            #SMU.write(":SYST:GRO (@1,2)")
            SMU.write(f":OUTP{channel} ON")
            SMU.write(f":OUTP{other_channel} ON")        
            log.info(f"GENERAL_CHANNEL_PULSING: BOTH Outputs ON, ACQUIRING from channel {mode}")
            SMU.write(f":INIT (@1,2)")

            # To stop code while acquiring
            self.sync(SMU)
            log.info(f"GENERAL_CHANNEL_PULSING: Data acquired. Fetching...")
            
            SMU.write(f":TRAC{channel}:DATA?")
            #SMU.write(f":FETC:ARR? (@{channel})")
            values = SMU.read()
            #SMU.write(":SYST:GRO (@1),(@2)")
            log.info(values)

        
        # TIME CURR STAT
        
        currents = []
        timestamps = []
        datadict = []
        
        # Assume that data is all comma-separated
        
        data_split = values.split(',')
        
        
        for i in range(len(data_split)):
            
            #print(f"Current value is i = {i}")
            # current first then time, due to convention from b2912a.
            
            if np.mod(i,2) == 0:
                currents.append(data_split[i])
        
            elif np.mod(i,2) == 1:
                timestamps.append(data_split[i])
            else:
                # np.mod(i,3) == 2
                pass
                #datadict['Status'].append(data_split[i])
        #print (timestamps, currents)
        
        for i in range(len(voltage_list)):
            
            datadict.append((timestamps[i], voltage_list[i], currents[i]))
        
        print(datadict)
        
        datadict = np.array(datadict)
        
        SMU.close()
        
        return datadict  
    
class Keithley707B:
    """Wrapper for the Keithley 707B switch matrix using slot/row/column channels.

    Default routes encoded here:
    - Keithley 2450: column 1 -> row A, column 3 -> row B
    - Keysight SMU: column 2 -> row A, column 4 -> row B

    Channel strings are formatted as `slot + row + column`, for example:
    - slot 1, row A, column 1 -> `1A01`
    - slot 1, row B, column 4 -> `1B04`
    """

    DEFAULT_SLOT = 1
    ROUTE_MAP = {
        "keithley": (("A", 1), ("B", 3)),
        "keysight": (("A", 2), ("B", 4)),
    }

    def __init__(self, device_no=0, address=None, slot=DEFAULT_SLOT):
        rm = pyvisa.ResourceManager()
        address_list = list(rm.list_resources())
        self.address = address if address is not None else address_list[device_no]
        self.rm = rm
        self.slot = int(slot)

    def _open_device(self):
        dev = self.rm.open_resource(self.address)
        dev.timeout = 10000
        dev.write_termination = "\n"
        dev.read_termination = "\n"
        return dev

    def build_channel(self, row, column, slot=None):
        slot = self.slot if slot is None else int(slot)
        row = str(row).strip().upper()
        column = int(column)
        return f"{slot}{row}{column:02d}"

    def _normalize_channel(self, channel):
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

    def _format_channels(self, channels):
        if isinstance(channels, list):
            return ",".join(self._normalize_channel(channel) for channel in channels)
        return self._normalize_channel(channels)

    def write_tsp(self, command):
        dev = self._open_device()
        try:
            dev.write(command)
        finally:
            dev.close()

    def query_tsp(self, expression):
        dev = self._open_device()
        try:
            dev.write(f"print({expression})")
            return dev.read().strip()
        finally:
            dev.close()

    def close_channel(self, channel):
        chs = self._format_channels(channel)
        self.write_tsp(f'channel.close("{chs}")')

    def open_channel(self, channel):
        chs = self._format_channels(channel)
        self.write_tsp(f'channel.open("{chs}")')

    def open_all(self):
        self.write_tsp('channel.open("allslots")')

    def get_closed_channels(self):
        return self.query_tsp('channel.getclose("allslots")')

    def get_route_channels(self, route_name, slot=None):
        route_key = route_name.lower()
        if route_key not in self.ROUTE_MAP:
            raise ValueError(f"Unknown 707B route '{route_name}'")
        return [
            self.build_channel(row=row, column=column, slot=slot)
            for row, column in self.ROUTE_MAP[route_key]
        ]

    def connect_named_route(self, route_name, slot=None):
        route_key = route_name.lower()
        channels = self.get_route_channels(route_key, slot=slot)
        if route_key == "keysight":
            self.open_all()
        self.close_channel(channels)
        return channels

    def connect_keithley_smu(self, slot=None):
        return self.connect_named_route("keithley", slot=slot)

    def connect_keysight_smu(self, slot=None):
        return self.connect_named_route("keysight", slot=slot)

    def reset(self):
        try:
            self.write_tsp("reset()")
        except Exception:
            pass


class KeithleySMU():
    
    """
    Wrapper for the Keithley 2450 using PyMeasure.
    Provides high-level functions for I-V sweeps and pulsed measurements.
    """

    def __init__(self, device_no=0, address=None, switch=None, switch_channel=None, connect_switch=False):
        """
        Initialize connection to Keithley SMU.
        Automatically grabs resource if not provided.
        """
        rm = pyvisa.ResourceManager()
        address_list = list(rm.list_resources())
        self.address = address if address is not None else address_list[device_no]

        self.smu = Keithley2450(self.address)
        # optional switch matrix and channel assignment
        self.switch = switch
        self.switch_channel = switch_channel
        self.switch_profile = "keithley"
        if connect_switch and self.switch is not None:
            try:
                self.connect_switch_path()
                print("Switch path prepared for KeithleySMU")
            except Exception as e:
                print(f"Warning: could not connect switch channel {self.switch_channel}: {e}")

        
        #self.smu.reset()
        self.smu.use_rear_terminals()
        print("Active terminals:", self.smu.ask(":ROUT:TERM?").strip())
        self.smu.apply_voltage()
        self.smu.measure_current()

        print("Devices found:", self.address)
        print("Handshaking with:", self.smu.id)

        # Load tiny IV resistance probe pattern
        self.tiny_IV = (
            "C:/Users/amdm/OneDrive - Nanyang Technological University/"
            "ViPSA data folder/sweep patterns/tinyIV.csv"
        )
        self.resistance_df = pd.read_csv(self.tiny_IV)

    def connect_switch_path(self):
        if self.switch is None:
            return

        route = self.switch_channel if self.switch_channel is not None else self.switch_profile
        route_name = route.lower() if isinstance(route, str) else None

        if hasattr(self.switch, "connect_named_route") and route_name in {"keithley", "keysight"}:
            self.switch.connect_named_route(route_name)
            return

        self.switch.close_channel(route)

    def disconnect_switch_path(self):
        if self.switch is None:
            return

        if hasattr(self.switch, "open_all"):
            self.switch.open_all()
            return

        if self.switch_channel is not None:
            self.switch.open_channel(self.switch_channel)

    def close_session(self):
        try:
            self.abort_measurement()
        except Exception:
            pass

        try:
            self.disconnect_switch_path()
        except Exception:
            pass

        try:
            adapter = getattr(self.smu, "adapter", None)
            connection = getattr(adapter, "connection", None)
            if connection is not None:
                connection.close()
        except Exception:
            pass

    #------------------------- Utility -----------------------------

    def get_address(self):
        return self.address

    def reset_device(self):
        self.smu.reset()
        self.smu.apply_voltage()
        self.smu.measure_current()
        
    def write(self, command):
        self.smu.write(command)
        
    def ask(self, command):
        response = self.smu.ask(command)
        return response

    def prepare_contact_probe(self, voltage, compliance):
        self.connect_switch_path()
        self.smu.write(":ROUT:TERM REAR")
        self.smu.apply_voltage()
        self.smu.measure_current()
        self.smu.compliance_current = compliance
        self.smu.write(f":SENS:CURR:PROT {compliance}")
        self.smu.write(f":SOUR:VOLT:LEV {voltage}")
        self.smu.write(":OUTP ON")

    def stop_output(self):
        self.smu.write(":OUTP OFF")

    def abort_measurement(self):
        try:
            self.smu.write(":ABOR")
        except Exception:
            pass
        try:
            self.smu.write(":OUTP OFF")
        except Exception:
            pass
        
        
    #------------------------- Quick probe -------------------------

    def get_contact_current(self, voltage, compliance=0.1, nplc=1.0, settle=0.02, adr=None):
        """
        Source a DC voltage and read current (absolute).
        Parameters
        ----------
        voltage : float (V)
        compliance : float (A)  # make this high enough so you don't current-limit by accident
        nplc : float            # integration time; 1 PLC ≈ 20 ms @ 50 Hz
        settle : float (s)      # wait after setting voltage
        """
        try:
            # configure
            #self.smu.reset()
            self.smu.write(":ROUT:TERM REAR")
            self.smu.apply_voltage()
            self.smu.measure_current()
            self.smu.compliance_current = compliance
            # fast-ish vs precise: tune NPLC
            self.smu.write(":SENS:CURR:NPLC {:.6f}".format(nplc))
            # turn output on explicitly and set voltage
            self.smu.write(":OUTP ON")
            self.smu.source_voltage = float(voltage)
            if settle > 0:
                time.sleep(settle)
            i = abs(float(self.smu.current))
            print(f"Contact Current : {i} A ; Resistance : {abs(float(voltage))/i}")
            # leave output off afterward
            self.smu.write(":OUTP OFF")
            return i
        except Exception as e:
            print("Error measuring contact current:", e)
            return None
        
    def get_contact_current_fast(self, voltage, settle=0.02):
        """
        Lightning-fast absolute current read for repeated probing loops.
        Assumes SMU already configured in voltage-source / current-measure mode.
    
        Args:
            voltage : float (V)
                Voltage to apply.
            settle : float (s)
                Wait after setting voltage before reading.
    
        Returns:
            float : absolute current in A
        """
        try:
            # Just set voltage and read current, nothing else
            self.smu.source_voltage = float(voltage)
            if settle > 0:
                time.sleep(settle)
            i = abs(float(self.smu.current))
            return i
        except Exception as e:
            print("Error in get_contact_current_fast:", e)
            return 0.0

    def run_read_probe(self, read_vlist, compliance=1e-3, delay=1e-3, nplc=0.01, label="probe",
                       progress_callback=None, stream_chunk=25):
        """
        Simple read-probe: step through a small voltage list and record current.
        Uses PyMeasure properties (no manual SCPI), so it's safe and consistent.
    
        Parameters
        ----------
        read_vlist : list of floats
            Voltages to apply for the probe (e.g. from tinyIV.csv).
        compliance : float, default 1e-6 A
            Current compliance limit.
        delay : float, default 1 ms
            Time to wait at each step (s).
        nplc : float, default 0.01
            Integration time in power line cycles.
        label : str, default "probe"
            Tag for verbose printing.
    
        Returns
        -------
        np.array of (timestamp, voltage, current)
        """
        print(f"  ↪ Running {label}: {len(read_vlist)} points, "
              f"compliance={compliance} A, delay={delay} s, NPLC={nplc}")
    
        self.smu.compliance_current = compliance
        self.smu.write(f":SENS:CURR:NPLC {nplc}")
    
        data = []
        pending = []
        t0 = time.perf_counter()
        for v in read_vlist:
            self.smu.source_voltage = float(v)
            time.sleep(delay)
            i = float(self.smu.current)
            ts = time.perf_counter() - t0
            point = (ts, float(v), i)
            data.append(point)
            pending.append(point)
            if callable(progress_callback) and len(pending) >= stream_chunk:
                progress_callback(pending)
                pending = []

        if callable(progress_callback) and pending:
            progress_callback(pending)
    
        return np.array(data)

    #------------------------- Segment detection -------------------

    def identify_linear_segments(self, voltages, times, tol=1e-9, delay_input=None):
        """
        Break a voltage list into regions of constant step.
        Returns list of dicts with vstart, vstop, vstep, delay, type.
        """
        segments = []
        start_idx = 0
        while start_idx < len(voltages) - 1:
            step = voltages[start_idx + 1] - voltages[start_idx]
            if delay_input is not None:
                delay = delay_input
            else:
                delay = times[start_idx + 1] - times[start_idx]
            
            idx = start_idx + 1
            while idx < len(voltages) - 1:
                this_step = voltages[idx + 1] - voltages[idx]
                if abs(this_step - step) > tol:
                    break
                idx += 1

            vstart = voltages[start_idx]
            vstop = voltages[idx]
            seg_type = "hold" if abs(step) < tol else "linear"

            segments.append(
                {
                    "vstart": vstart,
                    "vstop": vstop,
                    "vstep": step,
                    "delay": delay,
                    "type": seg_type,
                }
            )
            start_idx = idx
        return segments

    def split_by_polarity(self, voltages):
        """
        Split a sweep into cycles of positive and negative polarity.
        Returns list of (cycle_no, tag, vlist).
        """
        cycles = []
        cycle_no = 0
        current = []
        tag = "p" if voltages[0] >= 0 else "n"
        for v in voltages:
            if v >= 0 and tag == "n":
                cycle_no += 1
                cycles.append((cycle_no, tag, current))
                current = [v]
                tag = "p"
            elif v < 0 and tag == "p":
                cycle_no += 1
                cycles.append((cycle_no, tag, current))
                current = [v]
                tag = "n"
            else:
                current.append(v)
        if current:
            cycle_no += 1
            cycles.append((cycle_no, tag, current))
        return cycles
    
    #------------------------- 4way split ----------------------------

    def split_sweep_by_4(self, voltages, vtol=1e-9):
        """
        Robust 4-way split using turning points + explicit zeros.
        Works even if cycle amplitudes change (e.g., forming has higher +Vmax).
        Returns: list of (seg_no, tag, vlist, idx0, idx1)
        """
    
        v = np.asarray(voltages, dtype=float)
        n = len(v)
        if n < 3:
            return []
    
        dv = np.diff(v)
    
        # Sign of dv, but "carry forward" through plateaus (dv==0) so peaks are detected
        dv_s = np.sign(dv)
        for i in range(len(dv_s)):
            if dv_s[i] == 0:
                dv_s[i] = dv_s[i - 1] if i > 0 else 0
        if dv_s[0] == 0:
            nz = np.nonzero(dv_s)[0]
            if len(nz):
                dv_s[0] = dv_s[nz[0]]
    
        # Turning points: slope sign flips
        turn_idxs = (np.where(dv_s[1:] * dv_s[:-1] < 0)[0] + 1).tolist()
    
        # Explicit ~0 points (your sweep has real zeros at boundaries)
        zero_idxs = np.where(np.abs(v) <= vtol)[0].tolist()
    
        # Breakpoints = start + turning points + zeros + end
        bps = sorted(set([0] + turn_idxs + zero_idxs + [n - 1]))
    
        segments = []
        seg_no = 0
    
        for a, b in zip(bps[:-1], bps[1:]):
            if b <= a:
                continue
    
            vseg = v[a:b + 1]
            # skip segments that are only zeros
            nz = np.where(np.abs(vseg) > vtol)[0]
            if len(nz) == 0:
                continue
    
            v0 = float(vseg[nz[0]])
            v1 = float(vseg[nz[-1]])
    
            polarity = "p" if v0 >= 0 else "n"
            if polarity == "p":
                tag = "pf" if v1 >= v0 else "pb"
            else:
                tag = "nf" if v1 <= v0 else "nb"
    
            seg_no += 1
            segments.append((seg_no, tag, vseg.tolist(), a, b))
    
        return segments


    def list_IV_sweep_split_4(
        self,
        csv_path,
        compliance_pf,
        compliance_pb,
        compliance_nf,
        compliance_nb,
        delay=None,
        nplc=0.01,
        wait_time=0.0,
        progress_callback=None,
                    ):
            """
            Keithley/PyMeasure version of 4-way split sweep.
            Returns (data_array, resistance_array) as np.arrays of (t, v, i).
            """
            self.connect_switch_path()
        
            df = pd.read_csv(csv_path, dtype=float)
            times = df.iloc[:, 0].astype(float).to_numpy()
            volts = df.iloc[:, 1].astype(float).to_numpy()
        
            # tiny probe list
            vlist_resistance = self.resistance_df.iloc[:, 1].astype(float).tolist()
        
            # Instrument init (keep what you already use)
            self.smu.write("*RST")
            self.smu.write(":ROUT:TERM REAR")
            self.smu.write(f":SENS:CURR:NPLC {nplc}")
            self.smu.write(":OUTP ON")
        
            # Split sweep into pf/pb/nf/nb segments
            splits = self.split_sweep_by_4(volts.tolist())
        
            # Compliance + measurement range maps (tune these!)
            comp_map = {
                "pf": compliance_pf,
                "pb": compliance_pb,
                "nf": compliance_nf,
                "nb": compliance_nb,
            }
        
            # Key idea: range should be based on expected CURRENT during that segment,
            # not just compliance. Start conservative, then tune.
            meas_range_map = {
                "pf": max(1e-6, compliance_pf * 10),
                "pb": max(1e-6, compliance_pb * 10),
                "nf": max(1e-6, compliance_nf * 10),
                "nb": max(1e-9, min(1e-6, compliance_nb * 10)),  # often HRS read lives here
            }
        
            data_array = []
            resistance_array = []
        
            for seg_no, tag, vlist_seg, idx0, idx1 in splits:
                if wait_time:
                    time.sleep(wait_time)
        
                comp = comp_map[tag]
        
                # IMPORTANT: set a measurement range that can actually resolve small currents
                # (PyMeasure maps this to a Keithley range command under the hood)
                self.smu.current_range = meas_range_map[tag]
                self.smu.compliance_current = comp
        
                print(f"\nSeg {seg_no}: {tag} | points={len(vlist_seg)} | comp={comp} A | I_range={self.smu.current_range} A")
        
                # Optional: state-aware probe after finishing a polarity return-to-zero segment
                # This mimics your Keysight "pb/nb" probe behavior.
                if tag in ("pb", "nb"):
                    probe_comp = 1e-6 if tag == "pb" else 1e-5
                    probe = self.run_read_probe(
                        vlist_resistance,
                        compliance=probe_comp,
                        delay=1e-5,
                        nplc=nplc,
                        label="HRS read-probe" if tag == "pb" else "LRS read-probe",
                        progress_callback=progress_callback,
                    )
                    resistance_array.extend(probe)
        
                # Now execute this segment as one “linear segment”
                # Use your existing identify_linear_segments to respect step size timing if needed
                times_seg = times[idx0:idx1+1].tolist()
        
                segments = self.identify_linear_segments(
                    vlist_seg,
                    times_seg,
                    delay_input=delay
                )
        
                for s in segments:
                    chunk = self.run_linear_segment(
                        s,
                        compliance=comp,
                        delay_input=delay,
                        progress_callback=progress_callback,
                    )
                    data_array.extend(chunk)
        
            self.smu.write(":OUTP OFF")
            self.disconnect_switch_path()
            return np.array(data_array), np.array(resistance_array)

    #------------------------- Core sweep ----------------------------

    def run_linear_segment(self, seg, compliance, delay_input = None, progress_callback=None, stream_chunk=25):
        """
        Execute one sweep/hold segment on the Keithley.
        Returns list of (time, voltage, current).
        """
        vstart, vstop, vstep, delay, seg_type = (
            seg["vstart"],
            seg["vstop"],
            seg["vstep"],
            seg["delay"],
            seg["type"],
        )
        
        if delay_input is not None : 
            delay = delay_input
        self.smu.current_range = compliance*10
        self.smu.compliance_current = compliance
        self.smu.source_voltage = vstart

        currents, voltages, timestamps = [], [], []
        pending = []
        t0 = time.perf_counter()

        if seg_type == "hold":
            # just hold vstart for N points
            npts = max(1, int(round((vstop - vstart) / delay))) if delay > 0 else 1
            for _ in range(npts):
                self.smu.source_voltage = vstart
                time.sleep(delay)
                current = self.smu.current
                now = time.perf_counter() - t0
                currents.append(current)
                voltages.append(vstart)
                timestamps.append(now)
                pending.append((now, vstart, current))
                if callable(progress_callback) and len(pending) >= stream_chunk:
                    progress_callback(pending)
                    pending = []
        else:  # linear
            for v in np.arange(vstart, vstop + vstep, vstep):
                self.smu.source_voltage = v
                time.sleep(delay)
                current = self.smu.current
                now = time.perf_counter() - t0
                currents.append(current)
                voltages.append(v)
                timestamps.append(now)
                pending.append((now, v, current))
                if callable(progress_callback) and len(pending) >= stream_chunk:
                    progress_callback(pending)
                    pending = []

        if callable(progress_callback) and pending:
            progress_callback(pending)

        return list(zip(timestamps, voltages, currents))

    def list_IV_sweep_split(
        self,
        csv_path,
        pos_compliance,
        neg_compliance,
        SMU_range=None,
        delay=None,
        acq_delay=None,
        adr=None,
        pos_channel=None,
        neg_channel=None,
        wait_time=None,
        progress_callback=None
    ):
        """
        Backward-compatible: returns (data_array, resistance_array)
        each as np.array of (timestamp, voltage, current).
        """
        self.connect_switch_path()
        df = pd.read_csv(csv_path, dtype=float)
        times = df.iloc[:, 0].tolist()
        volts = df.iloc[:, 1].tolist()
    
        cycles = self.split_by_polarity(volts)
        data_array = []
        resistance_array = []
    
        # tiny read list from your file (2nd column)
        vlist_resistance = self.resistance_df.iloc[:, 1].astype(float).tolist()
        self.smu.write("*RST")
        self.smu.write(":ROUT:TERM REAR")
        print("Active terminals:", self.smu.ask(":ROUT:TERM?").strip())
        self.smu.write(":OUTP ON")
        
        for cycle_no, tag, vlist in cycles:
            time.sleep(wait_time)
            
            comp = pos_compliance if tag == "p" else neg_compliance
            # align time window for this cycle
            idx0 = volts.index(vlist[0])
            idx1 = idx0 + len(vlist)
            times_cycle = times[idx0:idx1]
    
            segments = self.identify_linear_segments(vlist, times_cycle, delay_input = delay)
            
            #state-aware resistance probe (tiny read, gentle timing)
            probe_chunk = self.run_read_probe(vlist_resistance,
                                              compliance=1e-6 if tag== "p" else 1e-5,
                                              delay=1e-5,
                                              label="HRS read-probe" if tag== "p" else "LRS read-probe",
                                              progress_callback=progress_callback)

            resistance_array.extend(probe_chunk)

            print(f"\nCycle {cycle_no} ({tag}, compliance: {comp}): {len(segments)} segment(s)")
            for seg in segments:
                if seg["type"] == "linear":
                    print(f"  Linear: {seg['vstart']} → {seg['vstop']}"
                          f" step {seg['vstep']} V, delay {seg['delay']} s")
                else:
                    print(f"  Hold:   {seg['vstart']} V, delay {seg['delay']} s/pt")
    
            # run segments
            for seg in segments:
                sweep_chunk = self.run_linear_segment(
                    seg,
                    comp,
                    delay_input=delay,
                    progress_callback=progress_callback,
                )
                data_array.extend(sweep_chunk)

            
        
        self.smu.write(":OUTP OFF")
        self.disconnect_switch_path()
        return np.array(data_array), np.array(resistance_array)

    #------------------------- Pulsed ------------------------------

    def pulsed_measurement(
        self,
        csv_path,
        current_compliance,
        set_width=0.01,
        bare_list=None,
        set_acquire_delay=None,
        adr=None,
    ):
        """
        Pulsed measurement for Keithley 2450.
        Signature kept compatible with existing execution code.

        Returns
        -------
        np.ndarray
            Array of shape (N, 3):
            [relative_timestamp_s, source_voltage_V, measured_current_A]
        """
        self.connect_switch_path()
        if adr is None:
            adr = self.address

        if set_acquire_delay is None:
            set_acquire_delay = 0.6 * set_width

        if set_width <= 0:
            raise ValueError("set_width must be > 0")
        if set_acquire_delay < 0:
            raise ValueError("set_acquire_delay must be >= 0")
        if set_acquire_delay >= set_width:
            raise ValueError("set_acquire_delay must be smaller than set_width")

        # Build voltage list
        if bare_list is not None:
            vlist = [float(v) for v in bare_list]
        else:
            df = pd.read_csv(csv_path, dtype=float)
            vlist = df.iloc[:, 1].astype(float).tolist()

        if len(vlist) == 0:
            return np.empty((0, 3), dtype=float)

        rm = pyvisa.ResourceManager()
        smu = None

        try:
            smu = rm.open_resource(adr)
            smu.timeout = 10000
            smu.write_termination = "\n"
            smu.read_termination = "\n"

            # Basic 2450 setup
            smu.write("*RST")
            smu.write("*CLS")
            smu.write(":ROUT:TERM REAR")

            smu.write(':SOUR:FUNC VOLT')
            smu.write(':SENS:FUNC "CURR"')
            smu.write(f":SENS:CURR:PROT {float(current_compliance)}")

            # Fast-ish measurement settings
            smu.write(":SENS:CURR:RANG:AUTO ON")
            smu.write(":SENS:CURR:NPLC 0.01")

            # Keep format simple: current only
            smu.write(":FORM:ELEM CURR")

            smu.write(":OUTP ON")

            data = []
            t0 = time.perf_counter()
            next_start = t0

            for v in vlist:
                v = float(v)

                # Absolute-time scheduling to reduce drift
                while True:
                    now = time.perf_counter()
                    wait = next_start - now
                    if wait <= 0:
                        break
                    if wait > 0.002:
                        time.sleep(wait - 0.001)

                pulse_start = time.perf_counter()

                # Apply pulse
                smu.write(f":SOUR:VOLT {v}")

                # Wait to acquisition point
                acquire_time = pulse_start + set_acquire_delay
                while True:
                    now = time.perf_counter()
                    wait = acquire_time - now
                    if wait <= 0:
                        break
                    if wait > 0.002:
                        time.sleep(wait - 0.001)

                # READ? performs a measurement and returns current
                current = float(smu.query(":READ?").strip().split(",")[0])

                timestamp = time.perf_counter() - t0
                data.append((timestamp, v, current))

                # Return to zero after pulse
                smu.write(":SOUR:VOLT 0")

                # Wait until pulse width completes
                next_start = pulse_start + set_width

            smu.write(":OUTP OFF")
            return np.array(data, dtype=float)

        except Exception as e:
            print("Error during single pulse measurement:", e)
            return None

        finally:
            try:
                if smu is not None:
                    smu.write(":OUTP OFF")
                    smu.close()
            except Exception:
                pass
            self.disconnect_switch_path()
