# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:25:18 2020

@author: voide
"""
from datetime import datetime
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import mlab
import statistics
import scipy
from scipy import signal
from scipy import stats

t0 = datetime.now()
filename = '1s_sample_0.1s_ti'
# read pickle file
infile = open('..\\Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()


audio = df.Micro[85]  # fichier audio brut
Fs = 40000  # fréquence d'acquisition

# var = statistics.variance(audio)

L = len(audio)
NFFT = int(2 ** np.ceil(np.log2(abs(L))))
# freq = Fs / 2 * np.linspace(0, 1, NFFT//2)

T = 1.0 / Fs

########################################################################
# fft
xf = np.linspace(0.0, 1.0/(2.0*T), NFFT//2)
yf = np.fft.fft(audio, NFFT)/L
yf = 2*abs(yf[:NFFT//2])

# on néglige les x premières fréquences
xf = xf[20:]
yf = yf[20:]

fig = plt.figure(figsize=(10,10))
fig.add_subplot(211)
plt.title('FFT')
plt.plot(xf, yf)

########################################################################
# # pdf
# fig.add_subplot(312)
# plt.title('PDF')

# N=75
# n_samples = NFFT//2//N
# sumi = 0
# i = 0
# j = 0
# l = 0
# bar = []

# for j in range(N):
#     sumi = 0
#     m = 0
#     for i in range(n_samples):
#         sumi = sumi + yf[l]
#     for m in range(n_samples):
#             bar.append(sumi)
#     l = l + n_samples

# plt.plot(bar/sum(bar))

########################################################################
# CDF
cumsum = np.cumsum(yf)
fig.add_subplot(212)
plt.title('CDF')
plt.plot(xf, cumsum/sum(yf))

