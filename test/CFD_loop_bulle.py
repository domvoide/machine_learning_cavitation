# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:25:18 2020

@author: voide

test pour vérifier où se trouve la cavitation par bulle alpha = 6°
"""
from datetime import datetime
import pickle
import numpy as np
from matplotlib import pyplot as plt

# Paramètres
#############################################################################
datatype = 'Uni'  # modifier aussi ligne 57
#############################################################################

t0 = datetime.now() 

filename = '1s_sample_0.1s_ti'
# read pickle file
infile = open('..\\Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

sample = df.Uni

cavit = []
no_cavit = []
bulle = []

fig = plt.figure(figsize=(10,6))

def plotCDF(i, color):
    global xf
    audio = sample[i]  # fichier audio brut
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
        if df.Alpha[i] == 6:
            bulle.append(cumsumNorm)
        else:
            cavit.append(cumsumNorm)
    

for i in range(len(sample)):
    if df.Cavit[i] == 0:
        c = 'b'
    else:
        if df.Alpha[i] == 6:
            c = 'g'
        else:
            c = 'r'
    plotCDF(i, c)

plt.text(12500, 0.1, 'Red : cavitation par poche\nGreen : cavitation par bulle\nBlue : no cavitation', fontsize = 15)
plt.xlabel('Frequencies [Hz]')
plt.ylabel('CDF [-]')

sumcavit=np.zeros(len(cavit[0]))
for j in range(len(cavit)):
    sumcavit = sumcavit + cavit[j]
sumcavit = sumcavit / len(cavit) 

sumbulle=np.zeros(len(bulle[0]))
for j in range(len(bulle)):
    sumbulle = sumbulle + bulle[j]
sumbulle = sumbulle / len(bulle) 

sumnocavit=np.zeros(len(no_cavit[0]))
for j in range(len(no_cavit)):
    sumnocavit = sumnocavit + no_cavit[j]
sumnocavit = sumnocavit / len(no_cavit) 

plt.grid()
plt.minorticks_on()
plt.plot(xf, sumcavit , color='black')
plt.plot(xf, sumnocavit , color='black')
plt.plot(xf, sumbulle , color='black')

#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')