# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:16:56 2020

@author: voide
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv (r'data_acc_tri.csv', header=None)
data_acc_tri = df[0]
acc_tri = data_acc_tri - np.mean(data_acc_tri)
del df

y = data_acc_tri[2400000:2800000]


# FFT

L = len(y)
NFFT = int(2 ** np.ceil(np.log2(abs(L))))
Fs = 40000
freq = Fs / 2 * np.linspace(0,1,NFFT//2)

yf = np.fft.fft(y, NFFT)/L
amp = 2 * abs(yf[1:NFFT//2+1])

fig = plt.figure(figsize=(30,5))
plt.plot(freq, amp)
plt.title('FFT')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Amplitude [-]')

# Déterminer les 10 peak les plus importants
freqmax_index = abs(np.argsort(-amp[:NFFT//2])[:10])
freqmax = np.sort(freq[freqmax_index])[::-1]

