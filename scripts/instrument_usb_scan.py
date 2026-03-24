import logging
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_and_install_pyusb():
    """
    Checks if the pyusb library is installed and attempts to install it if it's not.
    """
    try:
        import usb.core  # Try to import the library
        logging.info("pyusb is already installed.")
        return True
    except ImportError:
        logging.warning("pyusb is not installed. Attempting to install...")
        try:
            # Use subprocess to run pip install
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyusb'])
            import usb.core # Check if install was successful
            logging.info("pyusb successfully installed.")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install pyusb: {e}")
            logging.error("Please install pyusb manually (pip install pyusb) and try again.")
            return False
        except ImportError:
            logging.error("Installation of pyusb appears to have failed.")
            logging.error("Please install pyusb manually (pip install pyusb) and try again.")
            return False

def get_instrument_info(device):
    """
    Retrieves basic information about a USB device.

    Args:
        device: A pyusb device object.

    Returns:
        dict: A dictionary containing the vendor ID, product ID, and manufacturer
              and product strings, or None if an error occurs.
    """
    import usb.core
    import usb.util
    try:
        device_info = {
            "vendor_id": device.idVendor,
            "product_id": device.idProduct,
            "manufacturer": usb.util.get_string(device, device.iManufacturer),
            "product": usb.util.get_string(device, device.iProduct),
        }
        return device_info
    except usb.core.USBError as e:
        logging.error(f"Error getting device info: {e}")
        return None
    except AttributeError:
        logging.error("Device does not have the required attributes.")
        return None

def identify_instrument(device_info):
    """
    Attempts to identify the type of instrument based on vendor and product information.
    This is a basic example and may require расширенный logic for specific instruments.

    Args:
        device_info (dict): A dictionary containing device information.

    Returns:
        str: A string describing the instrument type, or "Unknown" if not identified.
    """
    if not device_info:
        return "Unknown"

    vendor_id = device_info.get("vendor_id", 0)
    product_id = device_info.get("product_id", 0)
    manufacturer = device_info.get("manufacturer", "").lower()
    product = device_info.get("product", "").lower()

    # Add your instrument identification logic here.  Examples:
    if vendor_id == 0x0403 and product_id == 0x6001:  # Example: FTDI USB Serial Converter
        return "Serial Converter"
    elif "arduino" in product or "g-lab" in manufacturer:
        return "Arduino/G-Lab device"
    elif "ivium" in manufacturer.lower():
        return "Ivium Instrument" #Identifies Ivium
    elif "keysight" in manufacturer.lower() or "agilent" in manufacturer.lower():
        return "Keysight/Agilent Instrument"
    elif "tektronix" in manufacturer.lower():
        return "Tektronix Instrument"
    elif "fluke" in manufacturer.lower():
        return "Fluke Instrument"
    # Add more rules for other instrument types as needed

    return "Unknown Instrument"

def scan_usb_ports():
    """
    Scans all USB ports for connected instruments and prints information
    about any instruments found.
    """
    import usb.core

    # Find all connected devices
    devices = usb.core.find(find_all=True)

    if devices is None:
        logging.info("No USB devices found.")
        return

    for device in devices:
        # Get basic device information
        device_info = get_instrument_info(device)

        if device_info:
            # Attempt to identify the instrument
            instrument_type = identify_instrument(device_info)
            logging.info(f"Found Device: Manufacturer: {device_info['manufacturer']}, Product: {device_info['product']}, Type: {instrument_type}")
        else:
            logging.info("Found a USB device, but could not retrieve full information.")

if __name__ == "__main__":
    if check_and_install_pyusb():
        scan_usb_ports()
    else:
        logging.error("Failed to install pyusb.  Please install it manually and run the script again.")
