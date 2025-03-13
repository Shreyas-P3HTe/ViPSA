
"""
Shreyas' code.

This module holds the instrument class for tools from Keysight.
Methods are written in SCPI. 
Make sure your environment is set to Python 3.10 and the correct drivers from keysight website are downloaded and installed.

Classes:
    KeysightSMU()
"""


import time
import numpy as np
import pyvisa
import pandas as pd
import logging as log

class KeysightSMU():
    
    def __init__(self, device_no, address=None):
        
        rm = pyvisa.ResourceManager()
        address_list = list(rm.list_resources())
        self.address = address_list[device_no]
        
        print("Devices found :" , self.address)
        
        if self.address == None : 
            print("ERROR: device could not be found")
        
        SMU = rm.open_resource(self.address) 
        print("Handshaking with : ", SMU.query("*IDN?"),"at",self.address)
        SMU.close()
                  
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
        if adr == None:
            adr = self.address
        
        rm = pyvisa.ResourceManager()
        
        try:
            SMU = rm.open_resource(adr)
            SMU.write(f"{command}")
        
        except Exception as e:
            print("Error sending :",e)
            
    def get_contact_current(self, voltage, compliance=10e-6, adr=None):
        
        if adr == None :
            adr = self.address
        
        rm = pyvisa.ResourceManager()
        
        try :
            SMU = rm.open_resource(adr)
            # Reset and configure the SMU
            SMU.write('*RST')
            SMU.write('*CLS')
            
            SMU.write('SOUR:FUNC VOLT')  # Set source function to voltage
            SMU.write('SENS:FUNC "CURR"')  # Set sense function to current
            SMU.write(f'SOUR:VOLT:LEV {voltage}')  # Set the voltage level
            SMU.write(f'SENS:CURR:PROT {compliance}')  # Set a high current compliance to avoid damage
                
            SMU.write('OUTP ON') #turn on the output

            
            current = abs(float(SMU.query('MEAS:CURR?'))) 
            print(f": Contact_current = {current} A")
            
            return current
            
        except Exception as e:
            print("Error measuring current :",e)    
                            
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
    
    def split_dataframe(self, df):
        """
        Takes in a DataFrame and splits the second column into positive and negative cycles.
    
        Arguments :
            df : pandas DataFrame
                DataFrame containing voltage data (second column).
    
        Returns :
            voltage_data : List of Lists with split data containing [[cycle no, "p" or "n" tag, DataFrame_cycle]]
        """
        voltage_data = []
        cycle_no = 0
        current_cycle = pd.DataFrame()  # Initialize empty DataFrame
        current_tag = ""
    
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce') #convert to numeric. coerce will replace non numeric values with NaN.
        vlist = df.iloc[:, 1].tolist()  # Extract the second column as voltage list
    
        for i in range(len(vlist)):
            v = vlist[i]
            if pd.isna(v): #check if v is nan, skip if it is.
                continue
            row = df.iloc[i:i + 1]  # Get the current row as a DataFrame
    
            if i == 0:  # Initialize the first cycle
                if v >= 0:
                    current_tag = "p"
                else:
                    current_tag = "n"
                current_cycle = pd.concat([current_cycle, row])
                continue
    
            if v >= 0 and current_tag == "n":  # Transition from negative to positive
                cycle_no += 1
                voltage_data.append([cycle_no, current_tag, current_cycle])
                current_tag = "p"
                current_cycle = row.copy()  # Start a new DataFrame
            elif v < 0 and current_tag == "p":  # Transition from positive to negative
                cycle_no += 1
                voltage_data.append([cycle_no, current_tag, current_cycle])
                current_tag = "n"
                current_cycle = row.copy()  # Start a new DataFrame
            else:  # Continue current cycle
                current_cycle = pd.concat([current_cycle, row])
    
        # Append the last cycle
        if not current_cycle.empty:
            cycle_no += 1
            voltage_data.append([cycle_no, current_tag, current_cycle])
    
        return voltage_data
    
    def simple_IV_sweep(self, vstart, vstop, vstep, compliance, delay, adr = None):
        
        #simple quick IV sweeping function. Address is assigned to self.address if 
        # by default. User can change this by passing it as an argument
        
        if adr == None:
            adr = self.address
        
        rm = pyvisa.ResourceManager()
        
        try:
            SMU = rm.open_resource(adr)
            # Example setup for a simple I-V sweep
            SMU.write("*RST")
            SMU.write("*CLS")
            SMU.write("SOUR:FUNC VOLT")  # Set source function to voltage
            SMU.write(f"SOUR:VOLT:START {vstart}")  # Set start voltage
            SMU.write(f"SOUR:VOLT:STOP {vstop}")  # Set stop voltage
            SMU.write(f"SOUR:VOLT:STEP {vstep}")  # Set voltage step
            SMU.write(f"SENS:CURR:PROT {compliance}")  # Set current compliance
            SMU.write("OUTP ON")
            SMU.write("SOUR:VOLT:MODE SWE")  # Set to sweep mode
            SMU.write(f"TRIG:DEL {delay}")  # Set trigger delay
            
            SMU.write("TRAC:POIN 1000")  # Set the number of points to store in the buffer
            SMU.write("TRAC:CLE")  # Clear the buffer
            SMU.write("TRAC:FEED SENS")  # Select the source of data to be stored in the buffer
            SMU.write("TRAC:FEED:CONT NEXT")  # Enable buffer to store readings
            
            # Initiate the sweep
            SMU.write("INIT")
            #SMU.write("FETC:ARR:CURR?")  # Fetch the array of currents measured
            time.sleep((vstop-vstart)/vstep*delay)
            # Wait for the sweep to complete
            SMU.query("*OPC?")
            
            # Retrieve the data from the buffer
            data = SMU.query("TRAC:DATA?")  # Fetch all the data points
            
            # Process the data (convert to float and split by comma)
            data = list(map(float, data.split(',')))
            
            # Turn off the output
            SMU.write("OUTP OFF")
                        
            SMU.close()
            
            return data
            
        except Exception as e:
            print("ERROR during IV sweep:", e)        
        
    def list_IV_sweep_manual(self, csv_path, pos_compliance, neg_compliance, delay=None, adr=None):
            
        # General purpose list scanning function. 
        # The CSV file for voltage lists can be generated using the listmaker app.
        # By default, the listmaker generated timing is used from CSV.
        # Delay time can be overwritten simply by passing it as an argument, the minimum delay is 50ms
        
        
        if adr == None:
            adr = self.address
        
        rm = pyvisa.ResourceManager()
        
        try :
            SMU = rm.open_resource(adr)
            
            df = pd.read_csv(csv_path)
            print(df)
            vlist = df.iloc[:, 1].tolist()  
            tlist = df.iloc[:, 0].tolist()
            
            SMU.write('*RST')
            SMU.write('*CLS')
            
            SMU.write('SOUR:FUNC VOLT')  # Set source function to voltage
            SMU.write('SENS:FUNC "CURR"')  # Set sense function to current
            
            # Prepare to store the data
            data = []
            column = 0
            # Perform the voltage sweep
            for voltage in vlist:
                
                # Set the appropriate compliance based on the voltage sign
                if voltage >= 0:
                    SMU.write(f'SENS:CURR:PROT {pos_compliance}')  # Set positive current compliance
                else:
                    SMU.write(f'SENS:CURR:PROT {neg_compliance}')  # Set negative current compliance

                # Set the voltage
                SMU.write(f'SOUR:VOLT {voltage}')
                # Wait for the source to settle
                SMU.query('*OPC?')
                # Measure the current
                current = SMU.query('MEAS:CURR?')
                
                if delay is None and column != len(vlist)+1:
                    timestamp = tlist[column]
                    time.sleep(tlist[column] - timestamp)
                    column += 1
                    
                elif delay <= 0.05:
                    delay = 0.05
                    timestamp = delay * column
                    column += 1
                    
                else :
                    timestamp = delay * column
                    column += 1
                    time.sleep(delay-0.05)
                # Store the voltage and current
                data.append((timestamp, voltage, current))


            SMU.close()
            
            # Convert the data to a numpy array for easy manipulation
            data = np.array(data)
            
            return data
        
        except Exception as e:
            print("ERROR during list sweep:", e) 

    def scan_list_divyam(self, smu, voltage_list, delay, acq_delay, current_compliance):
        
        try:
        
            voltage_string = str(voltage_list)
            voltage_string = voltage_string.replace('[','')
            voltage_string = voltage_string.replace(']','')
            pulse_no = len(voltage_list)
            print("here in copied code, ", pulse_no)
    
            #smu.timeout = 1000000  # 1000s
            smu.write_termination = '\n' # To define end of command.
            smu.read_termination = '\n' # To define end of command.
            smu.write('*CLS') # Clears the command queue.
            smu.write('*RST') # Resets the volatile memory.
            print(smu.query("SYST:ERR?"), "ONe")
            
            smu.write(":TRAC:FEED SENS") # The buffer stores measured data.
#            smu.write(":TRAC2:FEED SENS") # The buffer stores measured data.
            smu.write(":TRAC:FEED:CONT NEXT") # Make the buffer editable.
#            smu.write(":TRAC2:FEED:CONT NEXT")
            print(smu.query("SYST:ERR?"), "SEcond")
            
            smu.write(":TRAC:TST:FORM ABS") # Format of stored timestamps.
#            smu.write(":TRAC2:TST:FORM ABS") # Format of stored timestamps.
            smu.write(":FORM:ELEM:SENS TIME,VOLT,CURR") # Reading operation stores time, voltage, and current to the buffer.
            smu.write(":SOUR:FUNC:MODE VOLT") # Set the source to supply voltage.
            smu.write(":SOUR:VOLT:MODE LIST")
            print(smu.query("SYST:ERR?"), "Third")
            
            # curr = np.empty(1000, dtype=float) # The list of voltages to contruct a single pulse.
            # for i in range (1000): # To create an array of 0's.
            #     curr[i] = 0 
            # curr_string = ','.join(map(str, curr))
            # smu.write(f":SOUR:LIST:CURR {curr_string}")
            # print(smu.query("SYST:ERR?"), "this is before trigging")
            
            # smu.write(":SOUR2:FUNC:MODE VOLT") # Set the source to supply voltage.
            # smu.write(":SOUR2:VOLT:MODE LIST") # Supply voltage as a list.
            # smu.write(":SENS2:CURR:PROT {current_compliance}") # Set compliance current.
            smu.write(":SENS:CURR:PROT {current_compliance}") # Set compliance current.
            smu.write(":TRIG:COUN {pulse_no}") # Set the number of triggers.
            #smu.write(":TRIG2:COUN 1000") # Set the number of triggers.
            smu.write(":TRIG:SOUR TIMER") # Set the source of commands trigger as time. The trigger duration depends on time. 
            #smu.write(":TRIG2:SOUR TIMER") # Set the source of commands trigger as time. The trigger duration depends on time. 


            smu.write(":TRIG:TIM {delay}")
            #smu.write(":TRIG2:TIM {delay}")
            smu.write(":trig:acq:del {acq_delay}")
            #smu.write(":trig2:acq:del {acq_delay}")
            smu.write(":outp1 ON")
            #smu.write(":outp2 ON")
            print(smu.query("SYST:ERR?"))
            #bit_num = (4, 5, 6, 7, 8)
    
            # print(pulse_train_string)
            smu.write(f"SOUR:LIST:VOLT {voltage_string}") # Send the voltage list that was previously generated.
            print ("Running...")
            smu.write(":INIT (@1)")
            smu.query("*OPC?")
            print(smu.query("SYST:ERR?"))
            print ("Fetching...")
            values = smu.query(":FETCH:ARR? (@1)")
            #print(values)
            #input_string = smu.query(":FETCH:ARR? (@2)")
            smu.query("*OPC?")
            print(smu.query("SYST:ERR?"))
    
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
        
        except Exception as e :
            
            print("Error in sweeping single cycle:", e)

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
            print("values obtained")
        
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
       
    def list_IV_sweep_split(self, csv_path, pos_compliance, neg_compliance, set_range = None, 
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
        
        if set_range is None :
            set_range = 10e-3
        
        vlist = file_df.iloc[:,1].tolist()
        splits = self.split_list(vlist)

        try:

            rm = pyvisa.ResourceManager()
            SMU = rm.open_resource(adr)
            SMU.timeout = 30000000
            data_array = []

            for split in splits:

                compliance = (pos_compliance if split[1] == 'p' else neg_compliance)
                #channel = (pos_channel if split[1] == 'p' else neg_channel)  # channel swapping not enabled yet
                print("Cycle n0.:",split[0], " compliance :", compliance)
                vlist_cycle = split[2]
                print(vlist_cycle)
                data = self.scan_read_vlist(SMU, vlist_cycle, delay, acq_delay, compliance, set_range)
                # print(data)
                data_array.extend(data)
                # print(vlist_cycle,"\n", compliance, channel, delay, acq_delay )

            return data_array

        except Exception as e:
            print("ERROR during list sweep:", e)
    
    def pulsed_measurement(self, csv_path, current_compliance, set_width=0.01, bare_list = None, set_acquire_delay = None, adr = None):
        
        if adr == None:
            adr = self.address
            
        if set_acquire_delay == None :
            set_acquire_delay = set_width/2
        
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
                vlist_n.append(voltage)
                
        return vlist_p, vlist_n
            
        
        pass