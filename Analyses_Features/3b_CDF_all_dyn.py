# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:09:05 2020

@author: voide

Affichage des graphes des mesures dynamiques
CDF et PCA avec la mesure statique pour comparaison
enregistrement au format png sous analyse_feature

"""
import pickle
from sklearn.decomposition import PCA
import sounddevice as sd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

t0 = datetime.now() 

# Paramètres
################################################################################################
test_file = ['01', '02', '03', '04', '05', '06', '07', '08','09'] 
fCDF = np.arange(0, 21000, 1000)
Fs = 40000  # fréquence d'acquisition
fftstart = 20  # valeurs de fréquence à négliger au début
pathfig = 'C:\\Users\\voide\\Documents\\Master\\Machine Learning\
\Documents\\Analyse_features\\'
# Nombre de composants à prendre en compte pour la PCA
n_components = 2
# choix des PCA transformée à afficher sur les plots 2 et 3
datatype = 'Micro'
figsize = (12, 6)
###############################################################################################
# PCA statique pour servir de référence pour les mesures dynamiques
###############################################################################################

file_PCA_stat = 'PCA_micro'
infile = open('Datas\\Pickle\\' + file_PCA_stat + '.pckl', 'rb')
pca_stat = pickle.load(infile)
infile.close()

data  = pca_stat[0]
cav = pca_stat[1]

del pca_stat

pca_stat = PCA(n_components=n_components)
pca_stat.fit(data)
# mise en forme des std en fonction des composents choisis
X_stat = pca_stat.transform(data)


    
# sets de couleurs pour meilleure visualisation des points
color = plt.cm.tab20(np.linspace(0,1,18))





#############################################################################

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

for name in test_file:
    
    ############## plot points in transformed space ############################

    fig1 = plt.figure(figsize=figsize)
    ax1 = fig1.add_subplot(111)
    ax1.title.set_text('First two main components')
    lenptfct = int(len(X_stat)/18)
    for i,c in zip(range(0, len(X_stat), lenptfct), color):
        if cav[i] == 0:
            mkr = 'x'
            face = c
        else: 
            mkr = 'o'
            face ='none'
        ax1.scatter(X_stat[i:i+lenptfct, 0], X_stat[i:i+lenptfct, 1], edgecolors=c, 
                    marker=mkr, s=75,
                    facecolors=face, label= 'Pt : ' + str(int((i+lenptfct)/lenptfct)))
        
    # affichage des labels
    ax1.legend(loc='best', ncol=3)
    
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = 'Audio_dyn_' + name + '_1s_sample_0.1s_ti'
    # read pickle file
    infile = open('Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
    df = pickle.load(infile)
    infile.close()
    
    sample = df[datatype]  # feature à analyser
    
    
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
    c = np.linspace(0,len(sample),len(sample))
    colors = [plt.cm.rainbow(i) for i in np.linspace(0, 1, len(sample))]
    
    fig2 = plt.figure(figsize=figsize)
    ax = fig2.add_subplot(111)
    ax.set_title('CDF ' + name, fontsize=18)
    ax.set_xlabel('Frequencies [Hz]')
    ax.set_ylabel('CDF')
    ax.grid()
    ax.minorticks_on
    
    # trouver la valeure xf la plus proche des fréquences fCDF
    for i in range(len(fCDF)):
        argxf.append(find_nearest(xf,fCDF[i])[1])
    
    
    def plotCDF(i, c, Fs=Fs, L=L, NFFT=NFFT, T=T, xf=xf):
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
        plt.plot(xf, cumsumNorm , color=c)
        
        for k in range(len(fCDF)):
            data[fCDF[k]].append(cumsumNorm[argxf[k]])
    
    
    for i in range(len(sample)):
        sCDF = plotCDF(i, c=colors[i])
    # cbar = plt.colorbar(colors)
    # cbar.set_label('Sample number')
    
    plt.savefig(pathfig + 'CDF_dyn_'+ name + '_' + datatype +'.png')
    plt.close(fig2)
    
    # plot des features
    for j in range(len(fCDF)):
        x_feature.append(np.full(len(data[fCDF[j]]), xf[argxf[j]]))
    
    
    ########################## Plot CDF discrétisée ##############################
    
    data = pd.DataFrame(data).to_numpy()
    
   
    # pca = PCA(n_components=n_components)
    # pca.fit(data)
    # mise en forme des std en fonction des composents choisis
    X = pca_stat.transform(data)
    
    ############## plot points in transformed space ############################
    
    ax1.set_title('First two main components PCA dyn ' + name, fontsize=18)
    lenptfct = int(len(X)/12)
    
    # cmap = plt.cm.rainbow()
    sc = ax1.scatter(X[:, 0], X[:, 1], c=c, cmap='rainbow')
      
    cbar = plt.colorbar(sc)
    cbar.set_label('Sample number')
    ax1.set_xlabel('First component')
    ax1.set_ylabel('Second component')
    
    ax1.set_xlim(-1, 1.5)
    ax1.set_ylim(-0.5, 0.5)
    
    ax1.grid()
    ax1.minorticks_on
    
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(pathfig + 'PCA_dyn_'+ name + '_' + datatype +'.png')
    plt.close(fig1)
    
# fonction pour tester les fichier audios
def sound(track):
    sd.play(track, 44000)

    
#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')