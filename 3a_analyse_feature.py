# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 08:22:48 2020

@author: voide
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import statistics

# Paramètres
#############################################################################
datatype = 'Acc_X'  # modifier aussi ligne 56
filename = '1s_sample_0.1s_ti'
fCDF = [1500, 3500, 7000]
Fs = 40000  # fréquence d'acquisition
fftstart = 20  # valeurs de fréquence à négliger au début
pathfig = 'C:\\Users\\voide\\Documents\\Master\\Machine Learning\
\Documents\\Analyse_features\\'+ datatype +'\\'
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

# read pickle file
infile = open('..\\Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

##############################################################################
sample = df.Acc_X  # feature à analyser
##############################################################################

#initialisation des listes et dictionnaires
cavit = []      # vecteur CDF cavitant
no_cavit = []   # vecteur CDF non cavitant
bulle = []      # vecteur CDF bulle
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

################################# plot CDF ##################################

# création de la figure
fig = plt.figure(num=1, figsize=(10,6))
fig.title = ('CDF ' + datatype)
plt.title(fig.title)
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

# légende et labels pour le graphe
plt.text(12500, 0.1, 
         'Red : pocket cavitation\nBlue : no cavitation\nGreen : bubble cavitation \nX : features',
         fontsize = 15)
plt.xlabel('Frequencies [Hz]')
plt.ylabel('CDF [-]')

# moyenne pour les CDFs en cavitation
sumcavit=np.zeros(len(cavit[0]))
for j in range(len(cavit)):
    sumcavit = sumcavit + cavit[j]
meancavit = sumcavit / len(cavit) 

# moyenne pour les CDFs pas en cavitation par bulle
sumbulle=np.zeros(len(bulle[0]))
for j in range(len(bulle)):
    sumbulle = sumbulle + bulle[j]
meanbulle = sumbulle / len(bulle) 

# moyenne pour les CDFs pas en cavitation
sumnocavit=np.zeros(len(no_cavit[0]))
for j in range(len(no_cavit)):
    sumnocavit = sumnocavit + no_cavit[j]
meannocavit = sumnocavit / len(no_cavit) 

# plot des courbes moyennes
plt.plot(xf, meancavit , color='black')
plt.plot(xf, meannocavit , color='black')
plt.plot(xf, meanbulle , color='black')

# plot des features
for j in range(len(fCDF)):
    x_feature.append(np.full(len(data[fCDF[j]]), xf[argxf[j]]))
    plt.scatter(x_feature[j], data[fCDF[j]], color='black', marker='x', s=50,
            zorder=3)

plt.savefig(str(pathfig) + filename + '_' + fig.title + '.png', transparent=True)
############################# plot peak 2 peak ###############################

cavit = []
no_cavit = []
bulle = []

fig = plt.figure(2, figsize=(10,6))  # création figure
fig.title = ('Peak to peak (amplitude) ' + datatype)
plt.title(fig.title)
plt.grid()
plt.minorticks_on()

# Boucle pour plotter toutes les variances
for i in range(len(sample)):
    p2p = max(sample[i]) - min(sample[i])
    if df.Cavit[i] == 0:
        y = 0
        c = 'b'
        no_cavit.append(p2p)
    else:
        if df.Alpha[i] == 6:
            y =0.5
            c = 'g'
            bulle.append(p2p)
        else:
            y = 1
            c = 'r'
            cavit.append(p2p)
    plt.scatter(p2p, y, color=c)
        
    
# Mise en forme du graphique
plt.text(7, 0.2, 'Red : pocket cavitation\nBlue : no cavitation\nGreen : bubble cavitation \nX : mean', fontsize = 15)
plt.xlabel('Peak to peak [-]')
plt.ylabel('Cavitation [-]')

# moyenne des variances par label
mean_no_cavit = np.mean(no_cavit)
mean_cavit = np.mean(cavit)
mean_bulle = np.mean(bulle)

# plot des moyennes
plt.scatter(mean_no_cavit, 0, c='black', edgecolor='white', marker='x', s=100)
plt.scatter(mean_cavit, 1, c='black', edgecolor='white', marker='x', s=100)
plt.scatter(mean_bulle, 0.5, c='black', edgecolor='white', marker='x', s=100)

plt.savefig(str(pathfig) + filename + '_' + fig.title + '.png', transparent=True)

############################# plot RMS #######################################

cavit = []
no_cavit = []
bulle = []

fig = plt.figure(3, figsize=(10,6))  # création figure
fig.title = ('RMS ' + datatype)
plt.title(fig.title)
plt.grid()
plt.minorticks_on()

# Boucle pour plotter toutes les variances
for i in range(len(sample)):
    rms = np.sqrt(sum(n*n for n in sample[i])/len(sample[i]))
    if df.Cavit[i] == 0:
        y = 0
        c = 'b'
        no_cavit.append(rms)
    else:
        if df.Alpha[i] == 6:
            y = 0.5
            c = 'g'
            bulle.append(rms)
        else:
            y = 1
            c = 'r'
            cavit.append(rms)
    plt.scatter(rms, y, color=c)
    
# Mise en forme du graphique
plt.text(1.1, 0.2, 'Red : pocket cavitation\nBlue : no cavitation\nGreen : bubble cavitation \nX : mean', fontsize = 15)
plt.xlabel('Root mean square [-]')
plt.ylabel('Cavitation [-]')

# moyenne des variances par label
mean_no_cavit = np.mean(no_cavit)
mean_cavit = np.mean(cavit)
mean_bulle = np.mean(bulle)

# plot des moyennes
plt.scatter(mean_no_cavit, 0, c='black', edgecolor='white', marker='x', s=100)
plt.scatter(mean_cavit, 1, c='black', edgecolor='white', marker='x', s=100)
plt.scatter(mean_bulle, 0.5, c='black', edgecolor='white', marker='x', s=100)

plt.savefig(str(pathfig) + filename + '_' + fig.title + '.png', transparent=True)

############################# plot Variance ##################################

cavit = []
no_cavit = []
bulle = []

fig = plt.figure(4, figsize=(10,6))  # création figure
fig.title = ('Variance ' + datatype)
plt.title(fig.title)
plt.grid()
plt.minorticks_on()

# Boucle pour plotter toutes les variances
for i in range(len(sample)):
    var = statistics.variance(sample[i])
    if df.Cavit[i] == 0:
        y = 0        
        c = 'b'
        no_cavit.append(var)
    else:
        if df.Alpha[i] == 6:
            y = 0.5
            c = 'g'
            bulle.append(var)
        else:
            y = 1
            c = 'r'
            cavit.append(var)
    plt.scatter(var, y, color=c)
    
# Mise en forme du graphique
plt.text(1.6, 0.2, 'Red : pocket cavitation\nBlue : no cavitation\nGreen : bubble cavitation \nX : mean', fontsize = 15)
plt.xlabel('Variance [-]')
plt.ylabel('Cavitation [-]')

# moyenne des variances par label
mean_no_cavit = np.mean(no_cavit)
mean_cavit = np.mean(cavit)
mean_bulle = np.mean(bulle)

# plot des moyennes
plt.scatter(mean_no_cavit, 0, c='black', edgecolor='white', marker='x', s=100)
plt.scatter(mean_cavit, 1, c='black', edgecolor='white', marker='x', s=100)
plt.scatter(mean_bulle, 0.5, c='black', edgecolor='white', marker='x', s=100)

plt.savefig(str(pathfig) + filename + '_' + fig.title + '.png', transparent=True)

#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')

