# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:36:44 2020

@author: dominiqu.voide

Création d'un fichier de features et labels pouvant être utilisé pour le 
machine learning fichier modifié pour les mesures dynamiques
"""

import numpy as np
import statistics
import pickle
import pandas as pd
from scipy import signal
from datetime import datetime

# Paramètres
#############################################################################
Fs = 40000  # fréquence d'acquisition
fftstart = 20  # valeurs de fréquence à négliger au début
fCDF = np.arange(0, 21000, 1000)  # fréquences à analyser pour la CDF

datatype = 'Micro' # choix du capteur
#############################################################################

t0 = datetime.now() 
filename = '1s_sample_0.1s_ti'  # mesure dynamique

# read pickle file
infile = open('Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()
sample = df[datatype] # choix du vecteur à analyser


#initialisation des listes et dictionnaires

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
    Réalisation de la CDF de l'échantillon
    
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
    Vecteur de la CDF

    """
   
    ########################################################################
    # fft
    
    yf = np.fft.fft(array, NFFT)/L
    yf = 2*abs(yf[:NFFT//2])
    
    # on néglige les x premières fréquences
    yf = yf[fftstart:]
    
    
    ########################################################################
    # CDF
    
    # normalisation des valeurs
    cumsum = np.cumsum(yf)
    sumyf = sum(yf)
    cumsumNorm = cumsum/sumyf
    
    CDFa = []
    for j in range(len(fCDF)):
        CDFa.append(cumsumNorm[argxf[j]])

    return CDFa


CDF = []


# boucle pour traiter tous les fichiers
for i in range(len(sample)):
    CDF.append(cdf(sample[i]))



# création des vecteurs X et y
X = pd.DataFrame(CDF)
y = df.Cavit

# sauvegarde en pickle
g = open('Datas\\Pickle\\Features\\Features_' + datatype
         + '_' + filename + '.pckl', 'wb')
pickle.dump((X,y), g)
g.close()

#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')