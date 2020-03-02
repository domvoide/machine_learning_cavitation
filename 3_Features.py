# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:36:44 2020

@author: dominiqu.voide
"""

import numpy as np
from math import sqrt
import statistics
import pickle
import pandas as pd
from scipy import signal

filename = 'Short_1s_sample_0.9s_step'
# read pickle file
infile = open('Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

def rms(audio):
    return sqrt(sum(n*n for n in audio)/len(audio))

def variance(audio):
    return statistics.variance(audio)

def maxPSD(audio):
    freqs, psd = signal.welch(audio)
    maxPSD = max(psd)
    return maxPSD
    
    
def fftfreq(audio, Fs):
    L = len(audio)
    NFFT = int(2 ** np.ceil(np.log2(abs(L))))
    freq = Fs / 2 * np.linspace(0,1,NFFT//2)
    
    yf = np.fft.fft(audio, NFFT)/L
    amp = 2 * abs(yf[1:NFFT//2+1])
      
    # Déterminer les 10 peaks les plus importants
    freqmax_index = abs(np.argsort(-amp[100:NFFT//2])[:10]) # néglige les fréquences en dessous de 100 Hz
    freqmax = np.sort(freq[freqmax_index])[::-1]
    return freqmax
 
RMS = []
var = []
PSD = []
fft0 = []
fft1 = []
fft2 = []
fft3 = []
fft4 = []
fft5 = []
fft6 = []
fft7 = []
fft8 = []
fft9 = []

Fs = 40000  # Fréquence


for i in range(len(df.Micro)):
    RMS.append(rms(df.Micro[i]))
    var.append(variance(df.Micro[i]))
    PSD.append(maxPSD(df.Micro[i]))
    fft_mat = fftfreq(df.Micro[i], Fs)
    fft0.append(fft_mat[0])
    fft1.append(fft_mat[1])
    fft2.append(fft_mat[2])
    fft3.append(fft_mat[3])
    fft4.append(fft_mat[4])
    fft5.append(fft_mat[5])
    fft6.append(fft_mat[6])
    fft7.append(fft_mat[7])
    fft8.append(fft_mat[8])
    fft9.append(fft_mat[9])

data = {'RMS' : RMS,
        'Var' : var,
        'PSD' : PSD,
        'fft0' : fft0,
        'fft1' : fft1,
        'fft2' : fft2,
        'fft3' : fft3,
        'fft4' : fft4,
        'fft5' : fft5,
        'fft6' : fft6,
        'fft7' : fft7,
        'fft8' : fft8,
        'fft9' : fft9
        }

X = pd.DataFrame(data)
y = df.Cavit

g = open('Datas\\Pickle\\Features\\Features' + filename + '.pckl', 'wb')
pickle.dump((X,y), g)
g.close()