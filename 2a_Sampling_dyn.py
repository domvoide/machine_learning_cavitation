# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:36:44 2020

@author: dominiqu.voide

Création d'échantillons de taille dt avec durée de recouvrement ti
modification du fichier 2_sampling pour les mesures dynamiques
"""


import pickle
import pandas as pd
from datetime import datetime


# Paramètres
#############################################################################
dt = 1      # durée de l'échantillon en secondes
ti = 0    # durée de recouvrement en secondes (pas entre chaque échantillon)
filename = 'Mesures_Turbicav_alpha10_40kHz_test_03.csv'
start_cav = 44 # début de la cavitation en seconde 43, 46, 44
stop_cav = 94 # fin de la cavitation en seconde 99, 97, 94
#############################################################################

t0 = datetime.now()
# importation de la mesure dynamique
folder_path = 'Datas\\Mesures_dynamiques\\'

df = pd.read_csv (folder_path + filename, header=None)
columns = ['Time', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Uni_Y', 'Micro', 'Hydro', ]
df.columns = columns

# Acquisition de la fréquence
f = int(round(1 / (df.Time[1]-df.Time[0]), 1))

# Création d'un dictionnaire vide pour les échantillons d'analyses
data = {'Acc_X': [],
        'Acc_Y': [],
        'Acc_Z': [],
        'Micro': [],
        'Uni_Y': [],
        'Hydro': [],
        'Cavit': []}

size = int(dt * f)  # longueur de l'échantillon
step = int((dt-ti) * f)  # écart entre le début de chaque échantillon

# création des échantillons
for i in range(0, len(df.Micro), step):
    data['Acc_X'].append(df.Acc_X[i: i + size])
    data['Acc_Y'].append(df.Acc_Y[i: i + size])
    data['Acc_Z'].append(df.Acc_Z[i: i + size])
    data['Micro'].append(df.Micro[i: i + size])
    data['Uni_Y'].append(df.Uni_Y[i: i + size])
    data['Hydro'].append(df.Hydro[i: i + size])
    if df.Time[i] <= start_cav or df.Time[i] >= stop_cav:
        data['Cavit'].append(0)
    else:
        data['Cavit'].append(1)
        

# Création d'une liste des échantillons de la mauvaise taille
dellist = []
for k in range(len(data['Micro'])):
    if len(data['Micro'][k]) != size:
        dellist.append(k)
        
dellist = dellist[::-1]  # inversion du sens de la liste

# Suppression des échantillons de mauvaise taille
for nb in dellist:
    del data['Acc_X'][nb]
    del data['Acc_Y'][nb]
    del data['Acc_Z'][nb]
    del data['Micro'][nb]
    del data['Uni_Y'][nb]
    del data['Hydro'][nb]
    del data['Cavit'][nb]


del df  # suppression de la variable df

# transformation en dataframe
df = pd.DataFrame(data)
test_version = filename[-6] + filename[-5]
# enregistrement en pickle
g = open('Datas\\Pickle\\Sampling\\Dyn_'+ str(test_version) + '_' + str(dt) + 's_sample_' + str(ti) +
         's_ti.pckl', 'wb')
pickle.dump(df, g)
g.close()

# Contrôle que tous les échantillons soient bien de la même taille
lenmicro = []   
j = 0
for j in range(len(df.Micro)):
    lenmicro.append(len(df.Micro[j]))

print('longueur des échantillons identiques : ' + 
      str(min(lenmicro)==max(lenmicro)))

#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')