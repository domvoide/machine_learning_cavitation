# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:25:18 2020

@author: voide

plot de toutes les courbes CDF avec les couleurs en fonction des labels
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

# Paramètres
#############################################################################
datatype = 'Uni'  # modifier aussi ligne 57
#############################################################################

t0 = datetime.now() 

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

# entrée utilisateur
filename = '1s_sample_0.1s_ti'
fCDF = [1500, 3500, 7000]
Fs = 40000  # fréquence d'acquisition
fftstart = 20  # valeurs de fréquence à négliger au début

# read pickle file
infile = open('..\\Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

sample = df.Uni  # feature à analyser

#initialisation des listes et dictionnaires
cavit = []      # vecteur CDF cavitant
no_cavit = []   # vecteur CDF non cavitant
data = {}       # dict des features CDF aux fréquences fCDF
for key in fCDF:
    data[key] = []
argxf = []      # index des fCDF sur le vecteur xf
x_feature = []  # vecteur x pour ploter les features fCDF

# calcul du vecteur xf et de la longeur de la fft NFFT
L = len(sample[0])
NFFT = int(2 ** np.ceil(np.log2(abs(L))))
T = 1.0 / Fs
xf = np.linspace(0.0, 1.0/(2.0*T), NFFT//2)
xf = xf[fftstart:]

# trouver la valeure xf la plus proche des fréquences fCDF
for i in range(len(fCDF)):
    argxf.append(find_nearest(xf,fCDF[i])[1])

# création de la figure
fig = plt.figure(figsize=(10,6))
plt.title('CDF ' + datatype)
plt.grid()
plt.minorticks_on()

def plotCDF(i, color, Fs=Fs, L=L, NFFT=NFFT, T=T, xf=xf):
    """

    Parameters
    ----------
    i : int
        index du vecteur Micro.
    color : str
        Couleur selon cavitation ou non.
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
    audio = sample[i]  # fichier audio brut
   
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
    
    # plot de la CDF
    plt.plot(xf, cumsumNorm , color=color)
    
    # dat = dat.append({'frame': str(i), 'count':i},ignore_index=True)
    
    for k in range(len(fCDF)):
        data[fCDF[k]].append(cumsumNorm[argxf[k]])

    
    if df.Cavit[i] == 0:
        no_cavit.append(cumsumNorm)
    else:
        cavit.append(cumsumNorm)
    

for i in range(len(sample)):
    if df.Cavit[i] == 0:
        c = 'b'
    else:
        c = 'r'
    plotCDF(i, c)

# légende et labels pour le graphe
plt.text(12500, 0.1, 'Red : cavitation par poche\nBlue : no cavitation\nX : features',
         fontsize = 15)
plt.xlabel('Frequencies [Hz]')
plt.ylabel('CDF [-]')

# moyenne pour les CDFs en cavitation
sumcavit=np.zeros(len(cavit[0]))
for j in range(len(cavit)):
    sumcavit = sumcavit + cavit[j]
meancavit = sumcavit / len(cavit) 

# moyenne pour les CDFs pas en cavitation
sumnocavit=np.zeros(len(no_cavit[0]))
for j in range(len(no_cavit)):
    sumnocavit = sumnocavit + no_cavit[j]
meannocavit = sumnocavit / len(no_cavit) 

# plot des courbes moyennes
plt.plot(xf, meancavit , color='black')
plt.plot(xf, meannocavit , color='black')

# plot des features
for j in range(len(fCDF)):
    x_feature.append(np.full(len(data[fCDF[j]]), xf[argxf[j]]))
    plt.scatter(x_feature[j], data[fCDF[j]], color='black', marker='x', s=50,
            zorder=3)

#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')