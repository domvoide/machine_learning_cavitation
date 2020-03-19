# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:09:05 2020

@author: voide

Analyse PCA des features
"""
import pickle
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


# Paramètres
#############################################################################
datatype = 'Uni_Y'  # modifier aussi ligne 66
filename = '1s_sample_0.1s_ti'
fCDF = np.arange(0, 21000, 1000)
Fs = 40000  # fréquence d'acquisition
fftstart = 20  # valeurs de fréquence à négliger au début
pathfig = 'C:\\Users\\voide\\Documents\\Master\\Machine Learning\
\Documents\\Analyse_features\\'

n_components = 2


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
infile = open('Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

#########################################################################
sample = df[datatype]  # feature à analyser
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

########################## Plot CDF discrétisée ##############################

data = pd.DataFrame(data).to_numpy()

fig = plt.figure(2, figsize=(20, 13))
fig.suptitle('PCA '  + datatype, fontsize=24)
ax1 = fig.add_subplot(221)
ax1.title.set_text('CDF discrétisée')
ax1.grid()
ax1.minorticks_on()
ax1.plot(fCDF, np.transpose(data))
ax1.set_xlabel('Frequencies [Hz]')
ax1.set_ylabel('CDF [-]')

# Build a PCA model of the data
pca = PCA()
pca.fit(data)

# Plot the cumulative variance explained by the EV
ratio = pca.explained_variance_ratio_
# plt.figure(3, figsize=(10, 6))
ax2 = fig.add_subplot(222)
ax2.title.set_text('Variance / components')
ax2.plot(np.arange(1, len(ratio) + 1), np.cumsum(ratio))
ax2.set_xlabel('Numbers of Components')
ax2.set_ylabel('Cumulated variance explained by the components')

# Use only the first two components
pca = PCA(n_components=n_components)
pca.fit(data)
X = pca.transform(data)

# plot points in transformed space
# plt.figure(5, figsize=(10, 6))
color=plt.cm.rainbow(np.linspace(0,1,18))
ax3 = fig.add_subplot(223)
ax3.title.set_text('First two main components')
for i,c in zip(range(0, 1188, 66),  color):
    ax3.scatter(X[i:i+66, 0], X[i:i+66, 1],c=c, label= 'Pt : ' + str(int((i+66)/66)))
ax3.legend(loc='best', ncol=2)

# plt.plot(X[66:730, 0], X[365:730, 1], 'or')
# plt.plot(X[730:, 0], X[730:, 1], '.k')

# plot standard deviation location
std = np.sqrt(pca.explained_variance_)
# plt.plot(std[0], 0, 'og', label='std[0],0')
# plt.plot(-std[0], 0, 'ok', label='-std[0],0')
# plt.plot(0, std[1], 'or', label='0, std[1]')
# plt.plot(0, -std[1], 'om', label='0, -std[1]')
ax3.set_xlabel('First component')
ax3.set_ylabel('Second component')
ax3.grid()
ax3.minorticks_on

# plot the mean and the different axes of the PCA
# plt.figure(6, figsize=(10, 6))
ax4 = fig.add_subplot(224)
ax4.title.set_text('Mean and differents axes of the PCA')
ax4.plot(fCDF, pca.mean_, label='Mean')
# Transform the axes points into curves
y = pca.inverse_transform([std[0], 0])
ax4.plot(fCDF, y, 'g', label='1st component [+std]')
y = pca.inverse_transform([-std[0], 0])
ax4.plot(fCDF, y, 'k', label='1st component [-std]')
y = pca.inverse_transform([1, std[1]])
ax4.plot(fCDF, y, 'r', label='2nd component [std]')
y = pca.inverse_transform([1, -std[1]])
ax4.plot(fCDF, y, 'm', label='2nd component [-std]')
ax4.set_xlabel('Frequencies [Hz]')
ax4.set_ylabel('CDF')
ax4.legend(loc=0, ncol=2)
ax4.grid()
ax4.minorticks_on

fig.tight_layout(rect=[0, 0.03, 1, 0.95])


pdf = PdfPages(pathfig + 'PCA_' + datatype +'.pdf')
pdf.savefig(1)

#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')