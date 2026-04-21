"""
Compatibility aggregator for ViPSA hardware motion classes.

Import from the dedicated modules for implementation details:
`stage.py`, `light.py`, and `zaber.py`.
"""

from vipsa.hardware.light import Light
from vipsa.hardware.stage import Stage, stage
from vipsa.hardware.zaber import Zaber

__all__ = ["Light", "Stage", "Zaber", "stage"]
