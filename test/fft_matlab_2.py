# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:16:56 2020

@author: voide
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.fftpack

df = pd.read_csv (r'acc_tri.csv', header=None)
acc_tri = df[0]
del df

y = acc_tri[2400000:2800000]
y = np.asarray(y)
Fs = 40000
T = 1 / Fs
# FFT
N = len(y)

yf = scipy.fftpack.fft(y) 
xf = np.linspace(0.0, 1.0/(2.0*T), N//2) 


fig = plt.figure(figsize=(30,5))
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.title('FFT')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Amplitude [-]')

# # Déterminer les 10 peak les plus importants
# freqmax_index = abs(np.argsort(-amp[:NFFT//2])[:10])
# freqmax = np.sort(freq[freqmax_index])[::-1]

