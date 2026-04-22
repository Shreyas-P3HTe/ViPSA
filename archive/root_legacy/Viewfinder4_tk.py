import os
import sys
import tkinter as tk


WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
VIPSA_CLEAN_ROOT = os.path.join(WORKSPACE_DIR, "vipsa_clean")

if VIPSA_CLEAN_ROOT not in sys.path:
    sys.path.insert(0, VIPSA_CLEAN_ROOT)

from vipsa.gui.Viewfinder4_tk import VipsaGUI  # noqa: E402


if __name__ == "__main__":
    root = tk.Tk()
    app = VipsaGUI(root)
    root.mainloop()
