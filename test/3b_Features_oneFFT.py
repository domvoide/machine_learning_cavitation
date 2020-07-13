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
from scipy import stats
from datetime import datetime

t0 = datetime.now()

filename = '1s_sample_0s_ti'
# read pickle file
infile = open('..\\Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
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

def maxCDF(audio):
    # cdf = stats.norm.cdf(audio)
    maxCDF = max(audio)
    return maxCDF


def fftfreq(audio, Fs):
    L = len(audio)
    NFFT = int(2 ** np.ceil(np.log2(abs(L))))
    freq = Fs / 2 * np.linspace(0, 1, NFFT//2)

    yf = np.fft.fft(audio, NFFT)/L
    amp = 2 * abs(yf[1:NFFT//2+1])
    
    # Déterminer les 10 peaks les plus importants
    freqmax_index = abs(np.argsort(-amp[100:NFFT//2])[:1]) # néglige les fréquences en dessous de 100 Hz
    freqmax = freq[freqmax_index]
    return freqmax


RMS = []
var = []
PSD = []
CDF = []
fft0 = []

Fs = 40000  # Fréquence


for i in range(len(df.Micro)):
    RMS.append(rms(df.Micro[i]))
    var.append(variance(df.Micro[i]))
    PSD.append(maxPSD(df.Micro[i]))
    CDF.append(maxCDF(df.Micro[i]))
    fft_mat = fftfreq(df.Micro[i], Fs)
    fft0.append(fft_mat[0])


data = {'RMS': RMS,
        'Var': var,
        'PSD': PSD,
        'CDF': CDF,
        'fft0': fft0,
        }

X = pd.DataFrame(data)
y = df.Cavit

g = open('Datas\\Pickle\\Features\\Features_' + filename + '.pckl', 'wb')
pickle.dump((X, y), g)
g.close()
t1 = datetime.now()
deltat = (t1 - t0) / 1000
print('Temps écoulé' + str(deltat))