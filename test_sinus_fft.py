# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:14:33 2020

@author: voide

"""
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.fftpack 
from scipy.signal import find_peaks
from scipy.signal import argrelextrema


# Number of samplepoints 

N = 600 

# sample spacing 

T = 1.0 / 800.0 
x = np.linspace(0.0, N*T, N) 
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(200.0 * 2.0*np.pi*x)
yf = scipy.fftpack.fft(y) 
xf = np.linspace(0.0, 1.0/(2.0*T), N//2) 

fig = plt.figure()
ax  = fig.add_subplot(211)
ax.plot(x,y)
ax1 = fig.add_subplot(212)
ax1.plot(xf, 2.0/N * np.abs(yf[:N//2])) 
ax1.label_outer()
peaks, _ = find_peaks(yf, height=0.3)

freqmax_index = argrelextrema(yf[0:N//2], np.greater)[0]

freqmax = np.sort(xf[freqmax_index])[::-1]
for i in freqmax:
    print('Fréquences maximales : ' + str(i) + ' Hertz')
