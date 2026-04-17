'''
This is a class to control the multiplexer. It has functions to connect to the mux board 
to communicate with the arduino and ask it to switch the channels.

Props to SHIBI for developing the board and programming the arduino, and to me for writing this code.
Emphasis on SHIBI though. I'm merely a humble python guy.

- Shreyas

'''

import serial
import time

class Multiplexer():

    def connect_mux(self):
        """Connects to the Arduino on serial port COM3 with baudrate 9600.
    
        Returns:
            A serial object representing the connection to the Arduino.
        """
        try:
            self.ser = serial.Serial('COM3', 9600)
            print("Connected to Arduino on COM3")
            return self.ser
        except serial.SerialException as e:
            print(f"Error connecting to Arduino: {e}")
            return None

    def disconnect_mux(self):
        if self.ser is not None:
            self.ser.close()
            print("Disconnected from Arduino.")        

    def send_command(self, command, ser=None):
        """Sends a command to the Arduino and reads multiple lines of response.
    
        Args:
            ser: The serial object representing the connection to the Arduino.
            command: The command to send to the Arduino.
    
        Returns:
            A list of responses received from the Arduino.
        """
        if ser is None:
            ser = self.ser

        if ser is not None:
            ser.write(command.encode()) 
            print(f"Command sent: {command}")
    
            responses = []
            timeout = 2  # Adjust timeout as needed
            start_time = time.time()
    
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode().strip()
                    responses.append(line)
                    print(f"Arduino response: {line}")
    
                if time.time() - start_time > timeout:
                    break
    
            return responses
        else:
            print("Error: Not connected to Arduino.")
            return None
        
    def channel_to_command(self, ch1, ch2, ser=None):
        """
        Converts pins (1 to 16) for both the channels to commands and sends it.
        ----------
        ch1 : pin number on channel 1, decimal int
        ch2 : pin number on channel 2, decimal int

        Returns
        -------
        confirmation
        
        """
        if ser is None:
            ser = self.ser

        # create a list of pins
        hex_pin_list = ['0000','0001','0002','0004','0008',
                        '0010','0020','0040','0080',
                        '0100','0200','0400','0800',
                        '1000','2000','4000','8000'] # Please don't laugh at me
        
        #   The fuckall logic here is to convert from pin number to the hex, you
        #   have to convert to the 2^[pin_number-1] to Hex. e.g. to acess pin
        #   no. 5, you need to do 2^(5-1), i.e. 16, and convert that to hex, 
        #   which is 0010. Yeaah. I know.
        
        try:
            command = f"chn 0x{hex_pin_list[ch1]}, 0x{hex_pin_list[ch2]}"
            self.send_command(command, ser=ser)
            print("sending ", command)
            return True
        except Exception as e:
            print("Error", e, "happened")
            return False
