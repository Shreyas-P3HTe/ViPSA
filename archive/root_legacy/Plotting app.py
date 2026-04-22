# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:15:09 2025

@author: amdm
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QFileDialog, QSpinBox, QLabel, QHBoxLayout)
import matplotlib.cm as cm

class CyclePlotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cycle Plot Viewer')
        self.layout = QVBoxLayout()

        self.load_button = QPushButton('Load CSV')
        self.load_button.clicked.connect(self.load_csv)
        self.layout.addWidget(self.load_button)

        self.range_layout = QHBoxLayout()
        self.start_spin = QSpinBox()
        self.end_spin = QSpinBox()
        self.range_layout.addWidget(QLabel('Cycle Range:'))
        self.range_layout.addWidget(QLabel('Start'))
        self.range_layout.addWidget(self.start_spin)
        self.range_layout.addWidget(QLabel('End'))
        self.range_layout.addWidget(self.end_spin)
        self.layout.addLayout(self.range_layout)

        self.plot_button = QPushButton('Plot Selected Cycles')
        self.plot_button.clicked.connect(self.plot_selected_cycles)
        self.layout.addWidget(self.plot_button)

        self.canvas = FigureCanvas(plt.Figure())
        self.layout.addWidget(self.canvas)

        self.setLayout(self.layout)
        self.df = None
        self.cycles = []

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open CSV', '', 'CSV files (*.csv)')
        if not path:
            return
        self.df = pd.read_csv(path)
        self.df['Current (A)'] = self.df['Current (A)'].abs()
        self.cycles = self.extract_cycles(self.df)
        self.start_spin.setMaximum(len(self.cycles))
        self.end_spin.setMaximum(len(self.cycles))
        self.end_spin.setValue(len(self.cycles))

    def extract_cycles(self, df):
        start = 0
        cycles = []
        for i in range(1, len(df)):
            if df['Voltage (V)'].iloc[i-1] == 0 and df['Voltage (V)'].iloc[i] > 0:
                if start != i - 1:
                    cycles.append(df.iloc[start:i].reset_index(drop=True))
                start = i
        if start < len(df):
            cycles.append(df.iloc[start:].reset_index(drop=True))
        return cycles

    def split_by_4_indices(self, vlist):
        splits = []
        current_tag = ""
        start_idx = 0
        pmax = max(vlist)
        nmax = min(vlist)

        for i in range(1, len(vlist)):
            v_prev = vlist[i - 1]
            v_now = vlist[i]

            if current_tag == "":
                if v_now > 0:
                    current_tag = "pf"
                elif v_now < 0:
                    current_tag = "nf"
                start_idx = i - 1

            if current_tag == "pf" and v_now < pmax and v_now < v_prev:
                splits.append((current_tag, start_idx, i))
                current_tag = "pb"
                start_idx = i - 1
            elif current_tag == "pb" and v_now < 0 and v_now < v_prev:
                splits.append((current_tag, start_idx, i))
                current_tag = "nf"
                start_idx = i - 1
            elif current_tag == "nf" and v_now > nmax and v_now > v_prev:
                splits.append((current_tag, start_idx, i))
                current_tag = "nb"
                start_idx = i - 1
            elif current_tag == "nb" and v_now > 0 and v_now > v_prev:
                splits.append((current_tag, start_idx, i))
                current_tag = "pf"
                start_idx = i - 1

        if current_tag and start_idx < len(vlist) - 1:
            splits.append((current_tag, start_idx, len(vlist)))
        return splits

    def plot_selected_cycles(self):
        if self.df is None:
            return

        self.canvas.figure.clf()
        ax = self.canvas.figure.add_subplot(111)

        start = self.start_spin.value() - 1
        end = self.end_spin.value()
        colormap = cm.get_cmap('Set1')
        colors = [colormap(i / 11) for i in range(11)]

        for idx, cycle in enumerate(self.cycles[start:end]):
            vlist = cycle['Voltage (V)'].tolist()
            ilist = cycle['Current (A)'].tolist()
            segments = self.split_by_4_indices(vlist)

            for tag, s, e in segments:
                linestyle = '-' if tag in ['pf', 'nf'] else '--'
                ax.plot(vlist[s:e], ilist[s:e], linestyle=linestyle, color=colors[idx % len(colors)])

        ax.set_yscale('log')
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Current (log scale)')
        ax.grid(True)
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = CyclePlotApp()
    mainWin.show()
    sys.exit(app.exec_())
