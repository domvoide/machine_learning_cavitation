# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:09:05 2020

@author: voide

Analyse PCA des features
"""
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

# Paramètres
#############################################################################

filename = '1s_sample_0.1s_ti'
fCDF = np.arange(0, 21000, 1000)
Fs = 40000  # fréquence d'acquisition
fftstart = 20  # valeurs de fréquence à négliger au début
pathfig = 'C:\\Users\\voide\\Documents\\Master\\Machine Learning\
\Documents\\Analyse_features\\'

n_components = 2

name = 'Micro'

#############################################################################

t0 = datetime.now() 

if not os.path.isdir(pathfig + filename):
    os.makedirs(pathfig + filename)
    
pathfignew = pathfig + filename + '\\'

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
infile = open('Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

color1=plt.cm.tab20(np.linspace(0,0.5,10))
color2=plt.cm.tab20(np.linspace(0.5,1,10))

   
#########################################################################
sample = df[name]  # feature à analyser
#########################################################################

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
fig = plt.figure(1, figsize=(10,6))
plt.title('CDF ' + name)
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

########################## Plot CDF discrétisée ###########################

data = pd.DataFrame(data).to_numpy()

fig = plt.figure(2, figsize=(15, 9))
 
# Build a PCA model of the data
pca = PCA()
pca.fit(data)

# Use only the first two components
pca = PCA(n_components=n_components)
pca.fit(data)
X = pca.transform(data)

################### plot points in transformed space #####################
# plt.figure(5, figsize=(10, 6))

ax = fig.add_subplot(111)
# ax.title.set_text('First two main components')
lenptfct = int(len(X)/18)
for i,c in zip(range(0, int(len(df.Alpha)/2), lenptfct), color1):
    if df.Cavit[i] == 0:
        mkr = 'x'
        face = c
    else: 
        mkr = 'o'
        face ='none'
    ax.scatter(X[i:i+lenptfct, 0], X[i:i+lenptfct, 1], edgecolors=c, 
           marker=mkr, s=75,
           facecolors=face, label= 'Pt : ' + str(int((i+lenptfct)/lenptfct)))
    
for i,c in zip(range(int(len(df.Alpha)/2), len(df.Alpha), lenptfct), color2):
    if df.Cavit[i] == 0:
        mkr = 'x'
        face = c
    else: 
        mkr = 'o'
        face ='none'
    ax.scatter(X[i:i+lenptfct, 0], X[i:i+lenptfct, 1], edgecolors=c, 
           marker=mkr, s=75,
           facecolors=face, label= 'Pt : ' + str(int((i+lenptfct)/lenptfct)))

ax.legend(loc='best', ncol=2)

ax.set_xlabel('First component')
ax.set_ylabel('Second component')
ax.grid()
ax.minorticks_on

# Plot des fichiers dynamiques
test_file = ['03', '06', '07']
couleur = ['coral', 'black', 'darkgreen']
lab = ['Série 1', 'Série 2', 'Série 3']

for number, col, l in zip(test_file, couleur, lab):
    filename = 'Audio_dyn_' + number + '_1s_sample_0.1s_ti'
    # read pickle file
    infile = open('Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
    df = pickle.load(infile)
    infile.close()
    
    sample = df[name]  # feature à analyser
    
    
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
    
    
    def plotCDF(i, Fs=Fs, L=L, NFFT=NFFT, T=T, xf=xf):
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
        # plt.plot(xf, cumsumNorm , color='lightgrey')
        
        for k in range(len(fCDF)):
            data[fCDF[k]].append(cumsumNorm[argxf[k]])
    
    
    for i in range(len(sample)):
        plotCDF(i)
    
    
    # plot des features
    for j in range(len(fCDF)):
        x_feature.append(np.full(len(data[fCDF[j]]), xf[argxf[j]]))
    
    
    ########################## Plot CDF discrétisée ##############################
    
    data = pd.DataFrame(data).to_numpy()
        
    # pca = PCA(n_components=n_components)
    # pca.fit(data)
    # mise en forme des std en fonction des composents choisis
    X = pca.transform(data)
    
    ############## plot points in transformed space ############################
    
    lenptfct = int(len(X)/12)
    c = np.linspace(0,len(X),len(X))
    # cmap = plt.cm.rainbow()
    sc = ax.scatter(X[:, 0], X[:, 1], c=col, label=l)
      


    
    ax.set_xlim(-1, 1.5)
    ax.set_ylim(-0.5, 0.5)
    
    # ax1.grid()
    # ax1.minorticks_on
    
    
    ax.legend(loc='best', ncol=3)
    
    # fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    
    # plt.tight_layout()

    # plt.savefig(pathfig + 'PCA_dyn_'+ name + '_' + datatype +'.png')



#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')