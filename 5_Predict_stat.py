# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:22:41 2020

@author: voide

Prediction avec algorithme knn entrainé avec les mesures statiques
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import neighbors
from matplotlib import pyplot as plt
from datetime import datetime


# Paramètres
#############################################################################
knn_file = 'kNN_Micro_1s_sample_0s_ti'
mesure = 'Dyn_03'   # fichier à prédire

#############################################################################

t0 = datetime.now() 

# load the model from disk
knn = pickle.load(open('Datas\\Pickle\\kNN\\' + knn_file + '.pckl', 'rb'))


filename = 'Features_Micro_' + mesure +'_1s_sample_0s_ti' 
infile = open('Datas\\Pickle\\Features\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

X = df[0]   # matrice X contenant les features
y = df[1]   # vecteur y conteant les lables

y_pred = knn.predict(X) # test de prédiction
error = 1 - knn.score(X, y) # calcul de l'erreur de test

fig2 = plt.figure(figsize=(15,3))
ax2 = fig2.add_subplot(111)
ax2.step(np.arange(0, 120, 1), y_pred, label='Prediction')
ax2.step(np.arange(0, 120, 1), y, label='Label')

# ax2.set_title('Prédiction entrainement statique, ' + mesure + ', Erreur: '+
#               str(round(100 * error,2)) + '%')
ax2.set_xlabel('Secondes')
ax2.set_ylabel('Cavitation')
ax2.grid(which='major', linestyle='-', linewidth='0.5')
ax2.grid(which='minor', linestyle=':', linewidth='0.25', color='lightgray')
ax2.set_yticks([0, 1])
ax2.legend(loc='upper left')
plt.ylim(-0.5, 1.5)
ax2.minorticks_on()

#affichage du temps écoulé
t1 = datetime.now()
print('\nTemps écoulé : ' + str(t1-t0) + ' [h:m:s]')