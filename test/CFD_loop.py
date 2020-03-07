# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:25:18 2020

@author: voide

plot de toutes les courbes CDF avec les couleurs en fonction des labels
"""
from datetime import datetime
import pickle
import numpy as np
from matplotlib import pyplot as plt


t0 = datetime.now()
filename = '1s_sample_0.1s_ti'
# read pickle file
infile = open('..\\Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

cavit = []
no_cavit = []

fig = plt.figure(figsize=(10,6))

def plotCDF(i, color):
    audio = df.Micro[i]  # fichier audio brut
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
    
    
    ########################################################################
    # CDF
    cumsum = np.cumsum(yf)
    sumyf = sum(yf)
    cumsumNorm = cumsum/sumyf
    plt.title('CDF')
    plt.plot(xf, cumsumNorm , color=color)
    
    if df.Cavit[i] == 0:
        no_cavit.append(cumsumNorm)
    else:
        cavit.append(cumsumNorm)


for i in range(len(df.Micro)):
    if df.Cavit[i] == 0:
        c = 'b'
    else:
        c = 'r'
    plotCDF(i, c)

plt.text(15000, 0.1, 'Red : cavitation\nBlue : no cavitation', fontsize = 15)
plt.xlabel('Frequencies [Hz]')
plt.ylabel('Density [-]')

sumcavit=np.zeros(len(cavit[0]))
for j in range(len(cavit)):
    sumcavit = sumcavit + cavit[j]
 
lenmicro = []   
j = 0    
for j in range(len(df.Micro)):
    lenmicro.append(len(df.Micro[j]))