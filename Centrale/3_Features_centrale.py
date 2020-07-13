# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:36:44 2020

@author: dominiqu.voide

Création d'un fichier de features et label pouvant être utilisé pour le 
machine learning
"""

import numpy as np
import pickle
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib import cm

# Paramètres
#############################################################################

filename = 'Centrale'  # fichier sample a ouvrir

Fs = 10000  # fréquence d'acquisition
fftstart = 20  # valeurs de fréquence à négliger au début
fCDF = np.linspace(0, 6000, 21)  # fréquences à analyser pour la CDF

#############################################################################

t0 = datetime.now() 

# read pickle file
infile = open('Datas\\Pickle\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()
columns = df.columns


#initialisation des listes et dictionnaires

data = {}       # dict des features CDF aux fréquences fCDF
CDFa = []
argxf = []      # index des fCDF sur le vecteur xf
x_feature = []  # vecteur x pour ploter les features fCDF
for key in fCDF:
    data[key] = []


# calcul du vecteur xf et de la longeur de la fft NFFT
L = len(df[columns[0]])
NFFT = int(2 ** np.ceil(np.log2(abs(L))))
T = 1.0 / Fs
xf = np.linspace(0.0, 1.0/(2.0*T), NFFT//2)
xf = xf[fftstart:]

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(1,1,1)
colors = plt.cm.Accent(np.linspace(0, 1, num=8))



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


def cdf(array, Fs=Fs, L=L, NFFT=NFFT, T=T, xf=xf, color=None):
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
    
    # plot de la CDF
    ax1.plot(xf, cumsumNorm , color=c)
    
    CDFa = []
    for j in range(len(fCDF)):
        CDFa.append(cumsumNorm[argxf[j]])

    return CDFa
    
CDF = []


# boucle pour traiter tous les fichiers
for col in columns[0:14]:
    if col[0] == 'A':
        c = colors[0]
    elif col[0] == 'B':
        c = colors[1]
    elif col[0] == 'C':
        c = colors[2]
    elif col[0] == 'D':
        c = colors[7]
    elif col[0] == 'E':
        c = colors[4]
    elif col[0] == 'F':
        c = colors[5]
    elif col[0] == 'G':
        c = colors[6]
    else:
        c = colors[7]
        
    CDF.append(cdf(df[col], color=c))

    
ax1.legend(columns, loc='best', ncol=3)
ax1.set_title('CDF mesures en centrale')
ax1.set_ylabel('CDF')
ax1.set_xlabel('Fréquence [Hz]')

# création des vecteurs X et y
X = pd.DataFrame(CDF)


# sauvegarde en pickle
g = open('Datas\\Pickle\\Features\\Features_Centrale_statique.pckl', 'wb')
pickle.dump(X, g)
g.close()

#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')