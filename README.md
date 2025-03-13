# ViPSA

# ViPSA Components

This repository contains the components for the ViPSA project. The components are organized into several Python modules, each responsible for different functionalities such as GUI, vision processing, stage control, and data handling.

## Directory Structure

- **GUI.py**: Contains the graphical user interface for controlling the stages, camera, and lights.
- **Vision.py**: Handles vision processing tasks such as capturing images and overlaying data.
- **Viewfinder3.py**: Manages the connection and control of Arduino and Zaber stages, as well as the camera feed.
- **Source_Measure_Unit.py**: Provides classes and methods for interacting with Keysight Source Measure Units (SMUs).
- **Openflexture.py**: Contains classes for controlling the Openflexture stage and lights.
- **Main4.py**: Implements various methods for connecting equipment, detecting contact, centering pads, and running measurements.
- **Listmaker2.py**: Generates voltage sweep and pulsing data, and provides a GUI for visualizing and saving the data.
- **Datahandling.py**: Handles data saving, plotting, and processing for IV sweeps and pulse measurements.
- **b2912a.py**: Library for controlling the B2912A instrument.
- **Adaptive_t.py**: Script for adaptive testing using the ViPSA methods.

## Installation

To use these components, ensure you have the following dependencies installed:

- Python 3.10 or higher
- PySimpleGUI
- OpenCV
- threading
- zaber_motion
- pandas
- matplotlib
- pyvisa

You can install the required packages using pip:

```bash
pip install PySimpleGUI opencv-python-headless threading zaber-motion pandas matplotlib pyvisa
```

## Usage

1. **GUI**: Run the `GUI.py` script to launch the graphical user interface for controlling the stages, camera, and lights.
2. **Vision Processing**: Use the functions in `Vision.py` for capturing images and overlaying data.
3. **Stage Control**: Utilize the classes in `Viewfinder3.py` and `Openflexture.py` to manage the connection and control of Arduino and Zaber stages.
4. **Data Handling**: Use `Datahandling.py` for saving, plotting, and processing IV sweep and pulse measurement data.
5. **Adaptive Testing**: Run the `Adaptive_t.py` script for adaptive testing using the ViPSA methods.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request with your changes. Ensure your code follows the existing style and includes appropriate documentation and tests.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For any questions or issues, please contact the project maintainer at [email@example.com].

