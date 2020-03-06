#-*- coding: utf-8 -*-
"""
Created on Fri Feb 28 07:57:07 2020

@author: voide
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt

infile = open('Datas\\Pickle\\Sampling\\Short_1s_sample_0.9s_step.pckl', 'rb')
df = pickle.load(infile)
infile.close()

y = df.Micro[80]
rate = 40000
######################################################################

# FFT

L = len(y)
NFFT = int(2 ** np.ceil(np.log2(abs(L))))
Fs = rate
freq = Fs / 2 * np.linspace(0,1,NFFT//2)

yf = np.fft.fft(y, NFFT)/L
amp = 2 * abs(yf[1:NFFT//2+1])

fig = plt.figure(figsize=(30,5))
plt.plot(freq, amp)
plt.title('FFT')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Amplitude [-]')

# Déterminer les 10 peak les plus importants
# freqmax_index = abs(np.argsort(-amp[100:NFFT//2])[:10])
freqmax_index = np.where(yf == max(yf[0:N//2]))
freqmax = np.sort(freq[freqmax_index])[::-1]