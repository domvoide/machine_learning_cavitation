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
filename = '1s_sample_0.1s_ti'
# read pickle file
infile = open('Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()


fCDF = [1500, 3500, 7000]
Fs = 40000  # fréquence d'acquisition
fftstart = 20  # valeurs de fréquence à négliger au début

#initialisation des listes et dictionnaires
cavit = []      # vecteur CDF cavitant
no_cavit = []   # vecteur CDF non cavitant
data = {}       # dict des features CDF aux fréquences fCDF
for key in fCDF:
    data[key] = []
argxf = []      # index des fCDF sur le vecteur xf
x_feature = []  # vecteur x pour ploter les features fCDF

# calcul du vecteur xf et de la longeur de la fft NFFT
L = len(df.Micro[0])
NFFT = int(2 ** np.ceil(np.log2(abs(L))))
T = 1.0 / Fs
xf = np.linspace(0.0, 1.0/(2.0*T), NFFT//2)
xf = xf[fftstart:]

def find_nearest(array, value):
    """
    Trouve la valeur et l'index la plus proche d'une valeur 
    choisie dans un vecteur

    Parameters
    ----------
    array : array
        vecteur où trouver la valeur.
    value : int
        valeur à trouver.

    Returns
    -------
    TYPE
        DESCRIPTION.
    idx : TYPE
        DESCRIPTION.

    """
    n = [abs(i-value) for i in array]
    idx = n.index(min(n))
    return array[idx], idx

# trouver la valeure xf la plus proche des fréquences fCDF
for i in range(len(fCDF)):
    argxf.append(find_nearest(xf,fCDF[i])[1])


def cdf(audio, Fs=Fs, L=L, NFFT=NFFT, T=T, xf=xf):
    """

    Parameters
    ----------
    audio : array
        vecteur audio brut.
    Fs : int, optional
        Fréquence d'acquisition. The default is Fs.
    L : int, optional
        Longueur fichier audio. The default is L.
    NFFT : int, optional
        Puissance de 2 supérieur à L. The default is NFFT.
    T : float, optional
        période (1/Fs). The default is T.
    xf : array of float64, optional
        Vecteur x pour la fft et la CDF. The default is xf.

    Returns
    -------
    None.

    """
   
    ########################################################################
    # fft
    
    yf = np.fft.fft(audio, NFFT)/L
    yf = 2*abs(yf[:NFFT//2])
    
    # on néglige les x premières fréquences
    yf = yf[fftstart:]
    
    
    ########################################################################
    # CDF
    cumsum = np.cumsum(yf)
    sumyf = sum(yf)
    cumsumNorm = cumsum/sumyf

    CDF1 = cumsumNorm[argxf[0]]
    CDF2 = cumsumNorm[argxf[1]]
    CDF3 = cumsumNorm[argxf[2]]

    return CDF1, CDF2, CDF3

def variance(audio):
    return statistics.variance(audio)


def maxPSD(audio):
    freqs, psd = signal.welch(audio)
    maxPSD = max(psd)
    return maxPSD


var = []
CDF1 = []
CDF2 = []
CDF3 = []

# boucle pour analyser tous les fichiers audios
for i in range(len(df.Micro)):
    var.append(variance(df.Micro[i]))
    CDF1.append(cdf(df.Micro[i])[0])
    CDF2.append(cdf(df.Micro[i])[1])
    CDF3.append(cdf(df.Micro[i])[2])

# création du dictionnaire
data = {'Var': var,
        'CDF1': CDF1,
        'CDF2': CDF2,
        'CDF3': CDF3,
        }

# création des vecteurs X et y
X = pd.DataFrame(data)
y = df.Cavit

# sauvegarde en pickle
g = open('Datas\\Pickle\\Features\\Features_' + filename + '.pckl', 'wb')
pickle.dump((X, y), g)
g.close()
t1 = datetime.now()
deltat = (t1 - t0)
print('Temps écoulé: ' + str(deltat))