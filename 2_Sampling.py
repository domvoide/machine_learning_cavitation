# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:36:44 2020

@author: dominiqu.voide

Création d'échantillons de taille dt avec durée de recouvrement ti
"""

import io
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import sounddevice as sd

# Paramètres
#############################################################################
dt = 1  # durée de l'échantillon en secondes
ti = 0  # durée de recouvrement en secondes (pas entre chaque échantillon)
#############################################################################

# importation du pickle contenant les 18 fichiers de données
folder_path = 'Datas\\Pickle\\all_data.pckl'
buf = io.open(folder_path, 'rb', buffering=(128 * 2048))
dataframe = pickle.load(buf)
buf.close()
del buf
dataframe = dataframe.transpose()

# creation d'une matrice plus compact pour commencer (plus rapide à travailler)
df2 = dataframe[['Alpha', 'Sigma', 'Time', 'Micro', 'Cavit']].copy()
del dataframe

# Acquisition de la fréquence
f = round(1 / (df2.Time[0][1]-df2.Time[0][0]), 1)

# Création d'un dictionnaire vide pour les échantillons d'analyses
data = {'Alpha': [],
        'Sigma': [],
        'Micro': [],
        'Cavit': []}

size = int(dt * f)  # longueur de l'échantillon
step = int((1-ti) * f)  # écart entre le début de chaque échantillon

# création des échantillons
for j in range(18):
    for i in range(0, len(df2.Micro[j]), step):
        a = df2.Micro[j][i: i + size]
        a = a.astype('float64')
        data['Micro'].append(a)
        data['Cavit'].append(df2.Cavit[j])
        data['Alpha'].append(df2.Alpha[j])
        data['Sigma'].append(df2.Sigma[j])


# Création d'une liste des échantillons de la mauvaise taille
dellist = []
for k in range(len(data['Micro'])):
    if len(data['Micro'][k]) != size:
        dellist.append(k)
        
# dellist = dellist[::-1]  # inversion du sens de la liste
# # Suppression des échantillons de mauvaise taille
# for nb in dellist:
#     print(nb)
#     del data['Micro'][nb]
#     del data['Cavit'][nb]
#     del data['Alpha'][nb]
#     del data['Sigma'][nb]

# transformation en dataframe et mise en forme pour compacter
df = pd.DataFrame(data)
df.Cavit = df.Cavit.astype('uint8')
df.Alpha = df.Alpha.astype('float32')
df.Sigma = df.Sigma.astype('float32')

# Affichage du graphe des points de fonctionnements testés
fig, ax = plt.subplots(figsize=(11, 8))
scatter = ax.scatter(df2.Alpha, df2.Sigma, c=df2.Cavit, cmap='RdYlBu')
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Cavitation")
ax.add_artist(legend1)
plt.xlabel(r'$\alpha$ [°]')
plt.ylabel(r'$\sigma$ [-]')
plt.grid()
plt.show()

# enregistrement en pickle
g = open('Datas\\Pickle\\Sampling\\' + str(dt) + 's_sample_' + str(ti) +
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
