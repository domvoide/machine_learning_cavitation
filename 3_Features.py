# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:36:44 2020

@author: dominiqu.voide

Création d'un fichier de features et label pouvant être utilisé pour le 
machine learning
"""

import numpy as np
import statistics
import pickle
import pandas as pd
from scipy import signal
from datetime import datetime

# Paramètres
#############################################################################

filename = '1s_sample_0.1s_ti'  # fichier sample a ouvrir
Fs = 40000  # fréquence d'acquisition
fftstart = 20  # valeurs de fréquence à négliger au début
fCDF = [1500, 3500, 7000]  # fréquences à analyser pour la CDF
#datatype = 'Audio' # modifier la ligne 34-35 également (Micro, accéléromètre, ...)
datatype = 'Uni'
#############################################################################

t0 = datetime.now() 

# read pickle file
infile = open('Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()
sample = df.Uni # choix du vecteur à analyser


#initialisation des listes et dictionnaires
cavit = []      # vecteur CDF cavitant
no_cavit = []   # vecteur CDF non cavitant
data = {}       # dict des features CDF aux fréquences fCDF
argxf = []      # index des fCDF sur le vecteur xf
x_feature = []  # vecteur x pour ploter les features fCDF
for key in fCDF:
    data[key] = []


# calcul du vecteur xf et de la longeur de la fft NFFT
L = len(sample[0])
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


def cdf(array, Fs=Fs, L=L, NFFT=NFFT, T=T, xf=xf):
    """

    Parameters
    ----------
    array : array
        vecteur brut.
    Fs : int, optional
        Fréquence d'acquisition. The default is Fs.
    L : int, optional
        Longueur fichier. The default is L.
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
    
    yf = np.fft.fft(array, NFFT)/L
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

def variance(array):
    return statistics.variance(array)


def maxPSD(array):
    freqs, psd = signal.welch(array)
    maxPSD = max(psd)
    return maxPSD

def p2p(array):
    return max(sample[i]) - min(sample[i])

var = []
P2P = []
CDF1 = []
CDF2 = []
CDF3 = []

# boucle pour traiter tous les fichiers
for i in range(len(sample)):
    var.append(variance(sample[i]))
    P2P.append(p2p(sample[i]))
    CDF1.append(cdf(sample[i])[0])
    CDF2.append(cdf(sample[i])[1])
    CDF3.append(cdf(sample[i])[2])

# création du dictionnaire
data = {'Var': var,
        'P2P': P2P,
        'CDF1': CDF1,
        'CDF2': CDF2,
        'CDF3': CDF3,
        }

# création des vecteurs X et y
X = pd.DataFrame(data)
y = df.Cavit

# sauvegarde en pickle
g = open('Datas\\Pickle\\Features\\Features_' + datatype 
         + '_' + filename + '.pckl', 'wb')
pickle.dump((X, y), g)
g.close()

#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')