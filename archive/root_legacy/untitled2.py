# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 16:57:59 2025

@author: amdm
"""

# -*- coding: utf-8 -*-
"""
Enhanced Memristor Cycle Plot Viewer
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QCheckBox, 
                            QListWidget, QAbstractItemView, QFileDialog, QSpinBox, 
                            QLabel, QHBoxLayout, QGroupBox, QRadioButton, QLineEdit,
                            QSplitter)
from PyQt5.QtCore import Qt
import matplotlib
import numpy as np

class CyclePlotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cycle Plot Viewer')
        self.setGeometry(100, 100, 1400, 800)
        
        # Main layout with splitter
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel for controls
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(8)
        left_widget.setLayout(left_layout)
        left_widget.setMaximumWidth(320)
        
        # Load button
        self.load_button = QPushButton('Load CSV')
        self.load_button.clicked.connect(self.load_csv)
        left_layout.addWidget(self.load_button)
        
        # Cycle range group
        range_group = QGroupBox("Cycle Selection")
        range_layout = QVBoxLayout()
        range_layout.setSpacing(5)
        
        range_spin_layout = QHBoxLayout()
        self.start_spin = QSpinBox()
        self.end_spin = QSpinBox()
        range_spin_layout.addWidget(QLabel('Start:'))
        range_spin_layout.addWidget(self.start_spin)
        range_spin_layout.addWidget(QLabel('End:'))
        range_spin_layout.addWidget(self.end_spin)
        self.start_spin.valueChanged.connect(self.sync_selection)
        self.end_spin.valueChanged.connect(self.sync_selection)
        range_layout.addLayout(range_spin_layout)
        
        # Set/Reset checkboxes in horizontal layout
        setreset_layout = QHBoxLayout()
        self.set_checkbox = QCheckBox('Set')
        self.set_checkbox.setChecked(True)
        self.reset_checkbox = QCheckBox('Reset')
        self.reset_checkbox.setChecked(True)
        setreset_layout.addWidget(self.set_checkbox)
        setreset_layout.addWidget(self.reset_checkbox)
        setreset_layout.addStretch()
        range_layout.addLayout(setreset_layout)
        
        range_group.setLayout(range_layout)
        left_layout.addWidget(range_group)
        
        # Cycle list with scroll
        cycle_group = QGroupBox("Available Cycles")
        cycle_layout = QVBoxLayout()
        cycle_layout.setContentsMargins(5, 5, 5, 5)
        self.cycle_list = QListWidget()
        self.cycle_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.cycle_list.setMaximumHeight(180)
        cycle_layout.addWidget(self.cycle_list)
        cycle_group.setLayout(cycle_layout)
        left_layout.addWidget(cycle_group)
        
        # Axis scale options
        scale_group = QGroupBox("Y-Axis Scale")
        scale_layout = QHBoxLayout()
        self.linear_radio = QRadioButton("Linear")
        self.log_radio = QRadioButton("Log")
        self.linear_radio.setChecked(True)
        scale_layout.addWidget(self.linear_radio)
        scale_layout.addWidget(self.log_radio)
        scale_group.setLayout(scale_layout)
        left_layout.addWidget(scale_group)
        
        # Axis range controls
        axis_group = QGroupBox("Axis Limits")
        axis_layout = QVBoxLayout()
        axis_layout.setSpacing(5)
        
        # X-axis
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel('X-min:'))
        self.xmin_input = QLineEdit()
        self.xmin_input.setPlaceholderText('auto')
        x_layout.addWidget(self.xmin_input)
        x_layout.addWidget(QLabel('X-max:'))
        self.xmax_input = QLineEdit()
        self.xmax_input.setPlaceholderText('auto')
        x_layout.addWidget(self.xmax_input)
        axis_layout.addLayout(x_layout)
        
        # Y-axis
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel('Y-min:'))
        self.ymin_input = QLineEdit()
        self.ymin_input.setPlaceholderText('auto')
        y_layout.addWidget(self.ymin_input)
        y_layout.addWidget(QLabel('Y-max:'))
        self.ymax_input = QLineEdit()
        self.ymax_input.setPlaceholderText('auto')
        y_layout.addWidget(self.ymax_input)
        axis_layout.addLayout(y_layout)
        
        reset_axis_btn = QPushButton('Reset Axis Limits')
        reset_axis_btn.clicked.connect(self.reset_axis_limits)
        axis_layout.addWidget(reset_axis_btn)
        
        axis_group.setLayout(axis_layout)
        left_layout.addWidget(axis_group)
        
        # Plot button
        self.plot_button = QPushButton('Plot Selected Cycles')
        self.plot_button.clicked.connect(self.plot_selected_cycles)
        left_layout.addWidget(self.plot_button)
        
        # Save button
        self.save_button = QPushButton('Save Figure')
        self.save_button.clicked.connect(self.save_figure)
        left_layout.addWidget(self.save_button)
        
        left_layout.addStretch()
        
        # Right panel for plot
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(2)
        right_widget.setLayout(right_layout)
        
        # Create matplotlib figure with proper size
        self.figure = plt.Figure(figsize=(11, 8), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(700, 600)
        
        # Add navigation toolbar for zoom/pan
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        
        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
        # Data storage
        self.df = None
        self.cycles = []
        self.cycle_numbers = []

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open CSV', '', 'CSV files (*.csv)')
        if not path:
            return
        self.df = pd.read_csv(path)
        self.cycles = self.extract_cycles(self.df)
        
        self.cycle_numbers = sorted(set([c[0] for c in self.cycles]))
        
        self.cycle_list.clear()
        for n in self.cycle_numbers:
            self.cycle_list.addItem(str(n))
        
        # Update spin boxes and select all cycles by default
        self.start_spin.setMinimum(1)
        self.start_spin.setMaximum(len(self.cycle_numbers))
        self.start_spin.setValue(1)
        self.end_spin.setMinimum(1)
        self.end_spin.setMaximum(len(self.cycle_numbers))
        self.end_spin.setValue(len(self.cycle_numbers))
        
        # Select all items
        self.sync_selection()

    def sync_selection(self):
        start = self.start_spin.value()
        end = self.end_spin.value()
        self.cycle_list.clearSelection()
        for i in range(start-1, end):
            item = self.cycle_list.item(i)
            if item:
                item.setSelected(True)

    def extract_cycles(self, df):
        cycles = []
        grouped = df.groupby(["Cycle", "SetReset"])
        for (cycle_num, setreset), subset in grouped:
            subset = subset.reset_index(drop=True)
            if not subset.empty:
                cycles.append((cycle_num, setreset, subset))
        return cycles

    def split_set_segments(self, vlist):
        vmax = max(vlist)
        pf_end = vlist.index(vmax)
        return [
            ("pf", 0, pf_end+1),
            ("pb", pf_end, len(vlist))
        ]

    def split_reset_segments(self, vlist):
        vmin = min(vlist)
        nf_end = vlist.index(vmin)
        return [
            ("nf", 0, nf_end+1),
            ("nb", nf_end, len(vlist))
        ]

    def reset_axis_limits(self):
        self.xmin_input.clear()
        self.xmax_input.clear()
        self.ymin_input.clear()
        self.ymax_input.clear()
        self.plot_selected_cycles()

    def plot_selected_cycles(self):
        if self.df is None:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        selected_cycles = [int(i.text()) for i in self.cycle_list.selectedItems()]
        
        sweep_types = []
        if self.set_checkbox.isChecked():
            sweep_types.append("Set")
        if self.reset_checkbox.isChecked():
            sweep_types.append("Reset")
        
        colormap = matplotlib.colormaps['Set2']
        cycle_colors = {n: colormap(i % 9) for i, n in enumerate(selected_cycles)}
        
        # Check if log scale
        is_log = self.log_radio.isChecked()
        
        # Track which cycles are actually plotted
        plotted_cycles = set()
        
        for (cycle_num, setreset, cycle_df) in self.cycles:
            if cycle_num in selected_cycles and setreset in sweep_types:
                vlist = cycle_df['Voltage (V)'].tolist()
                ilist = cycle_df['Current (A)'].tolist()
                
                # Convert to mA
                ilist = [i * 1e3 for i in ilist]
                
                # Take absolute value for log scale
                if is_log:
                    ilist = [abs(i) for i in ilist]
                
                if setreset == "Set":
                    segments = self.split_set_segments(vlist)
                else:
                    segments = self.split_reset_segments(vlist)
                
                color = cycle_colors[cycle_num]
                
                for tag, s, e in segments:
                    linestyle = '-' if tag in ("pf", "nf") else '--'
                    # Only label once per cycle
                    label = f"{cycle_num}" if cycle_num not in plotted_cycles else None
                    ax.plot(vlist[s:e], ilist[s:e], linestyle=linestyle, color=color, 
                           label=label, linewidth=1.5)
                    plotted_cycles.add(cycle_num)
        
        # Set Y-axis scale
        if is_log:
            ax.set_yscale('log')
        else:
            ax.set_yscale('linear')
        
        ax.set_xlabel('Voltage (V)', fontsize=14, fontweight='bold', fontname='Arial')
        ax.set_ylabel('Current (mA)', fontsize=14, fontweight='bold', fontname='Arial')
        
        # Apply axis limits if specified
        try:
            if self.xmin_input.text():
                xmin = float(self.xmin_input.text())
                ax.set_xlim(left=xmin)
            if self.xmax_input.text():
                xmax = float(self.xmax_input.text())
                ax.set_xlim(right=xmax)
            if self.ymin_input.text():
                ymin = float(self.ymin_input.text())
                ax.set_ylim(bottom=ymin)
            if self.ymax_input.text():
                ymax = float(self.ymax_input.text())
                ax.set_ylim(top=ymax)
        except ValueError:
            pass
        
        ax.grid(True, alpha=0.3)
        
        # Create compact legend with cycle numbers only
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Determine number of columns based on number of cycles
            ncol = 1 if len(handles) <= 8 else 2 if len(handles) <= 16 else 3
            
            legend = ax.legend(handles, labels, title='Cycle', fontsize=9, 
                             bbox_to_anchor=(1.01, 1), loc='upper left', 
                             borderaxespad=0, ncol=ncol, framealpha=0.95,
                             title_fontsize=10)
        
        # Add textbox for line style explanation
        textstr = 'Line Style:\n━━ Forward\n- - - Reverse'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props, family='monospace')
        
        self.figure.tight_layout(rect=[0, 0, 0.88, 1])
        self.canvas.draw()

    def save_figure(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", 
                                                  "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All Files (*)", 
                                                  options=options)
        if filename:
            self.figure.savefig(filename, bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = CyclePlotApp()
    mainWin.show()
    sys.exit(app.exec_())