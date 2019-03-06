'''
@author:     Zhengguang Zhao
@copyright:  Copyright 2016-2019, Zhengguang Zhao.
@license:    MIT
@contact:    zg.zhao@outlook.com

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import time


class EventDetector:

    def __init__(self, data, wins, labels):
        
        self.data = data
        self.wins = wins
        self.labels = labels
        self.xmax = len(data)
        self.fig, self.ax = plt.subplots(figsize = (10,4))
        self.l = self.ax.plot(data, 'k-')

        self.count = 0

       
        
        ani = FuncAnimation(self.fig, self.update, frames=self.gen_ln, interval = 100,
                        init_func=self.init_plot)
        #ani.save('eventdetector.gif', writer='imagemagick', fps=30)
        plt.show()
        

    def init_plot(self):
        self.ax.set_xlim(0, self.xmax)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_xlabel('Sample')
        self.ax.set_ylabel('Normalized Amplitude')
        return self.l

    def gen_ln(self):
        for ind, val in enumerate(self.wins):
            new_line = [val, self.data[val]]
            label = self.labels[ind]
            yield (new_line, label, val)

    def update(self, newln):
        
        if self.count == 0:
            time.sleep(10)
        else:
            time.sleep(1)
        #self.ln.set_data(newln[0][0], newln[1])
        label = newln[1]
        span = newln[2]

        if len(self.ax.patches):
            self.ax.patches.clear()
        
        self.ax.axvspan(span[0],span[-1],facecolor = 'g', alpha = 0.2)
        
        if label == 1:
            self.ax.plot(newln[0][0], newln[0][1], 'r-', animated=False)
        if label == 2:
            self.ax.plot(newln[0][0], newln[0][1], 'b-', animated=False)
        self.count += 1

        
        

    
