# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:36:44 2020

@author: dominiqu.voide

Création d'échantillons de taille dt avec durée de recouvrement ti
et création d'un graphe de l'amplitude puis enregistrement du fichier audio 
correspondant pour détection auditive de la cavitation
"""
import pickle
import pandas as pd
from datetime import datetime
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from matplotlib import pyplot as plt

# Paramètres
#############################################################################
dt = 1      # durée de l'échantillon en secondes
ti = 0.1    # durée de recouvrement en secondes (pas entre chaque échantillon)

#############################################################################

t0 = datetime.now()
# importation de la mesure dynamique
input_path = 'Datas\\Mesures_dynamiques\\'
output_path = 'C:\\Users\\voide\\Documents\\Master\\Machine Learning\
\Documents\\Analyse_features\\Audio_dyn\\'
file_nb = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
for nb in file_nb:
    filename = 'Mesures_Turbicav_alpha10_40kHz_test_' + nb +'.csv'
    df = pd.read_csv (input_path + filename, header=None)
    columns = ['Time', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Uni_Y', 'Micro', 'Hydro', ]
    df.columns = columns
    
    # Acquisition de la fréquence
    f = int(round(1 / (df.Time[1]-df.Time[0]), 1))
    
    # Création d'un dictionnaire vide pour les échantillons d'analyses
    data = {'Micro': []}
    
    size = int(dt * f)  # longueur de l'échantillon
    step = int((1-ti) * f)  # écart entre le début de chaque échantillon
    
    # création des échantillons
    for i in range(0, len(df.Micro), step):
        data['Micro'].append(df.Micro[i: i + size])
            
    
    
    # Création d'une liste des échantillons de la mauvaise taille
    dellist = []
    for k in range(len(data['Micro'])):
        if len(data['Micro'][k]) != size:
            dellist.append(k)
            
    dellist = dellist[::-1]  # inversion du sens de la liste
    
    # Suppression des échantillons de mauvaise taille
    for nb in dellist:
        del data['Micro'][nb]
        
    dfsample = pd.DataFrame(data)    
    #création du fichier audio wav
    name = 'Audio_dyn_' + filename[-6] + filename[-5]
    m = np.max(np.abs(df.Micro))
    sigf32 = (df.Micro/m).astype(np.float32)
    write(output_path + name + '.wav', f, sigf32)
    
    fig1 = plt.figure(figsize=(20, 6))
    fig1.suptitle(name, fontsize=24)
    ax1 = fig1.add_subplot(111)
    ax1.plot(df.Time, df.Micro)
    ax1.set_ylabel('Amplitude')
    ax1.set_xlabel('Secondes')
    ax1.set_ylim(-1, 1)
    ax1.grid()
    fig1.tight_layout(rect=[0.00, 0.03, 1.00, 0.93])
    plt.savefig(output_path + name + '.png')
    
    g = open('Datas\\Pickle\\Sampling\\' + name + '_' + str(dt) + 's_sample_' + str(ti) +
         's_ti.pckl', 'wb')
    pickle.dump(dfsample, g)
    g.close()
    
#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')