# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:12:24 2020

@author: voide

Algorithme de machine learning kNN en entrainant à l'aide d'un fichier 
dynamique et en testant avec un autre fichier dyamique
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
mesure_dyn_train = 'Dyn_01'
mesure_dyn_test = 'Dyn_03'
n_neighbors = 10
#############################################################################

t0 = datetime.now() 
filename_dyn_train = 'Features_Micro_' + mesure_dyn_train +'_1s_sample_0s_ti' 
filename_dyn_test = 'Features_Micro_' + mesure_dyn_test +'_1s_sample_0s_ti' 
# importation des features et des labels

# read pickle file apprentissage
infile = open('Datas\\Pickle\\Features\\' + filename_dyn_train + '.pckl', 'rb')
df_train = pickle.load(infile)
infile.close()

X = df_train[0]   # matrice X contenant les features
y = df_train[1]   # vecteur y conteant les lables

# read pickle file mesure dynamique pour test
infile = open('Datas\\Pickle\\Features\\' + filename_dyn_test + '.pckl', 'rb')
df_test = pickle.load(infile)
infile.close()

X_dyn = df_test[0]  # matrice X contenant les features
y_dyn = df_test[1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
# 70% training and 30% test, random_state pour avoir toujours les mêmes données en train,
# stratify = y pour avoir le même poucentage de label que les sets

# création du modèle kNN
knn = KNeighborsClassifier(n_neighbors=n_neighbors) # p = 1 for manhattan distance
print('Facteur k = ' + str(n_neighbors) + ' et distance euclidienne')

# Train the model using the training sets
train = knn.fit(X_train, y_train)
errortrain = 1 - knn.score(X_train, y_train)  # calcul de l'erreur d'entrainement
y_pred = knn.predict(X_test) # test de prédiction
errortest = 1 - knn.score(X_test, y_test) # calcul de l'erreur de test

# affichage des résultats
print('Erreur train: %f' % errortrain)
print('Erreur test: %f' % errortest)
print("Accuracy:" + str(metrics.accuracy_score(y_test, y_pred)))


y_pred_dyn = knn.predict(X_dyn)
errordyn = 1 - knn.score(X_dyn, y_dyn)
fig2 = plt.figure(figsize=(15,3))
ax2 = fig2.add_subplot(111)
ax2.step(np.arange(0, 120, 1), y_pred_dyn, label='Prediction')
ax2.step(np.arange(0, 120, 1), y_dyn, label='Label')

ax2.set_title('Prédiction ' + mesure_dyn_test + ', ' + str(n_neighbors) + ' voisins')
ax2.set_xlabel('Secondes')
ax2.set_ylabel('Cavitation')
ax2.grid(which='major', linestyle='-', linewidth='0.5')
ax2.grid(which='minor', linestyle=':', linewidth='0.25', color='lightgray')
ax2.set_yticks([0, 1])
ax2.legend(loc='best')
plt.ylim(-0.5, 1.5)
ax2.minorticks_on()

#affichage du temps écoulé
t1 = datetime.now()
print('\nTemps écoulé : ' + str(t1-t0) + ' [h:m:s]')