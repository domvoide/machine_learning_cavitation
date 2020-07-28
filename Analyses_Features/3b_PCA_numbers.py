# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:09:05 2020

@author: voide

Analyse PCA des features
"""
import pickle
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


# Paramètres
#############################################################################
capteurs = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Micro', 'Uni_Y', 'Hydro']
filename = '1s_sample_0.1s_ti'
fCDF = np.arange(0, 21000, 1000)
Fs = 40000  # fréquence d'acquisition
fftstart = 20  # valeurs de fréquence à négliger au début
pathfig = 'C:\\Users\\voide\\Documents\\Master\\Machine Learning\
\Documents\\Analyse_features\\'

n_components = 2

# choix des PCA transformée à afficher sur les plots 2 et 3
for name in capteurs:
    datatype = name
    print(datatype)
    if datatype == 'Acc_X':
        ptfct = [(0.81,-0.0125), (0.33, 0.048), (-0.39, -0.0128), (-0.24, -0.005),
                 (-0.19, 0.075), (-0.3, -0.075)]
    elif datatype == 'Acc_Y':
        ptfct = [(0.98, 0), (0.84, -0.03), (0.3, 0.07), (-0.01, -0.12),
                 (-0.38, 0.125), (-0.55, 0.035)]
    elif datatype == 'Acc_Z':
        ptfct = [(0.98, -0.08), (0.4, -0.09), (0.18, 0.27), (-0.71, -0.08),
                 (-0.5, -0.03), (-0.5, -0.25)]
    elif datatype == 'Micro':
        ptfct = [(0.75,0.25), (0.55, -0.06), (0.2, 0.02), (-0.2, -0.02),
                 (-0.4, 0.07), (-0.68, 0.0)]
    elif datatype == 'Uni_Y':
        ptfct = [(0.35,0.28), (0, -0.15), (-0.4, 0.2), (0.67, 0.13),
                  (-0.25, 0.22), (-0.2, -0.2)]
    else:
        ptfct = [(-0.1, 0), (-0.06, 0), (0, 0), (0.05, 0),
                  (0.1, -0.025), (0.2, -0.025)]
    
    labelptfct = []
    alpha = 'a'
    for i in range(len(ptfct)): 
        labelptfct.append(alpha) 
        alpha = chr(ord(alpha) + 1) 
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
    
    # trouver la valeure xf la plus proche des fréquences fCDF
    for i in range(len(fCDF)):
        argxf.append(find_nearest(xf,fCDF[i])[1])
    
    # création de la figure
    fig1 = plt.figure(figsize=(10,6))
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
        
    # couleur en fonction du label
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
    
    fig2 = plt.figure(figsize=(20, 13))
    fig2.suptitle('PCA '  + datatype, fontsize=24)
    ax1 = fig2.add_subplot(221)
    ax1.title.set_text('CDF discrétisée')
    ax1.plot(fCDF, np.transpose(data), alpha=0.3, color='lightgrey')
    ax1.set_xlabel('Frequencies [Hz]')
    ax1.set_ylabel('CDF [-]')
    
    
    # choix du nombre de composents à prendre en compte
    pca = PCA(n_components=n_components)
    pca.fit(data)
    # mise en forme des std en fonction des composents choisis
    X = pca.transform(data)
    print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
# enregistrement de la PCA statique pour comparaison avec les données dynamiques
    g = open('Datas\\Pickle\\PCA_micro.pckl', 'wb')
    pickle.dump((data,df.Cavit), g)
    g.close()
    
    ############## plot points in transformed space ############################
    
    # sets de couleurs pour meilleure visualisation des points
    color = plt.cm.tab20(np.linspace(0,1,18))
    
    
    ax2 = fig2.add_subplot(222)
    ax2.title.set_text('First two main components')
    lenptfct = int(len(X)/18)
    for i,c in zip(range(0, len(X), lenptfct), color):
        if df.Cavit[i] == 0:
            mkr = 'x'
            face = c
        else: 
            mkr = 'o'
            face ='none'
        ax2.scatter(X[i:i+lenptfct, 0], X[i:i+lenptfct, 1], edgecolors=c, 
                    marker=mkr, s=75,
                    facecolors=face, label= 'Pt : ' + str(int((i+lenptfct)/lenptfct)))
        
    # affichage des labels
    ax2.legend(loc='best', ncol=2)
    ax2.set_xlabel('First component')
    ax2.set_ylabel('Second component')
    ax2.grid()
    ax2.minorticks_on
    
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    ##############################################################################
    ## affichage de la localisation des déviation en fonction des composants
    
    std = np.sqrt(pca.explained_variance_)
    
    ##############################################################################
    # plot the mean and the different axes of the PCA
    
    ax1.plot(fCDF, pca.mean_, c='C0', label='Mean', linewidth=2, zorder=10,
             path_effects=[pe.Stroke(linewidth=4, foreground='w'), pe.Normal()])
    # Transform the axes points into curves
    y = pca.inverse_transform([std[0], 0])
    ax1.plot(fCDF, y, 'g', label='1st component [+std]', linewidth=2, zorder=10,
             path_effects=[pe.Stroke(linewidth=4, foreground='w'), pe.Normal()])
    y = pca.inverse_transform([-std[0], 0])
    ax1.plot(fCDF, y, 'k', label='1st component [-std]', linewidth=2, zorder=10,
             path_effects=[pe.Stroke(linewidth=4, foreground='w'), pe.Normal()])
    y = pca.inverse_transform([0, std[1]]) # [1, std[1]] code exemple énergie
    ax1.plot(fCDF, y, 'r', label='2nd component [std]', linewidth=2, zorder=10,
             path_effects=[pe.Stroke(linewidth=4, foreground='w'), pe.Normal()])
    y = pca.inverse_transform([0, -std[1]]) # [1, -std[1]] code exemple énergie
    ax1.plot(fCDF, y, 'm', label='2nd component [-std]', linewidth=2, zorder=10,
             path_effects=[pe.Stroke(linewidth=4, foreground='w'), pe.Normal()])
    
    ax1.legend(loc=0, ncol=2)
    ax1.grid()
    ax1.minorticks_on
    
    ax3 = fig2.add_subplot(223)
    for i, txt in enumerate(labelptfct):
        ax2.text(ptfct[i][0], ptfct[i][1], str(txt),
                 bbox=dict(boxstyle='circle', alpha=0.5, facecolor='white'))   
        y = pca.inverse_transform([ptfct[i][0], ptfct[i][1]])
        ax3.plot(fCDF, y, label='Pt fct : ' + str(txt))
                     
    
    ax3.legend(loc=0, ncol=2)
    ax3.set_xlabel('Frequencies [Hz]')
    ax3.set_ylabel('CDF [-]')
    ax3.grid()
    ax3.minorticks_on
    
    ax4 = fig2.add_subplot(224)
    lenptfct = int(len(df.Alpha)/18)
    for i,c in zip(range(0, len(df.Alpha), lenptfct), color):
        if df.Cavit[i] == 0:
            mkr = 'x'
            face = c
        else: 
            mkr = 'o'
            face ='none'
    
        ax4.scatter(df.Alpha[i:i+lenptfct], df.Sigma[i:i+lenptfct],edgecolors=c, 
                   marker=mkr, s=75,
                   facecolors=face, label= 'Pt : ' + str(int((i+lenptfct)/lenptfct)))
    
    a = np.arange(1 , 16, 1)
    b = 0.003 * a ** 3 - 0.035*a**2 + 0.6231*a + 0.8694
    ac = np.arange(7, 10.3, 0.1)
    c = 0.125 * ac**2 -2.975 * ac + 19.225
    d = 0.0833 * a + 1
    ax4.plot(a, b, ac, c, a, d, c='k')
    
    ax4.text(6, 6.5,'No Cavitation', fontsize=14, 
            bbox=dict(boxstyle='round', alpha=0.5, facecolor='white'))
    ax4.text(2.75, 1.65,'Bubble Cavitation', fontsize=14, rotation=10,
            bbox=dict(boxstyle='round', alpha=0.5, facecolor='white'))
    ax4.text(11, 5.5,'Pocket Cavitation', fontsize=14,
            bbox=dict(boxstyle='round', alpha=0.5, facecolor='white'))
    ax4.text(7, 0.85,'Supercavitation', fontsize=14, rotation=3,
            bbox=dict(boxstyle='round', alpha=0.5, facecolor='white'))
    
    ax4.legend(loc='best', ncol=2, )
    ax4.set_xlabel(r'$\alpha$ [°]')
    ax4.set_ylabel(r'$\sigma$ [-]')
    ax4.grid()
    
    # plt.tight_layout()
    
    plt.savefig(pathfig + 'Points_fonctionnement_' + datatype +'.png')
    
    ### Plot the cumulative variance explained by the EV pour savoir le nombre 
    # de composants à prendre en compte pour la PCA
    # PCA sur tous les composents
    # pca = PCA()
    # pca.fit(data)
    
    # ratio = pca.explained_variance_ratio_
    # fig3 = plt.figure(figsize=(10, 6))
    # ax5 = fig3.add_subplot(111)
    # ax5.title.set_text('Variance / components')
    # ax5.plot(np.arange(1, len(ratio) + 1), np.cumsum(ratio))
    # ax5.set_xlabel('Numbers of Components')
    # ax5.set_ylabel('Cumulated variance explained by the components')

#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')