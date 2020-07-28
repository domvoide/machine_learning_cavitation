# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:12:24 2020

@author: voide

Algorithme de machine learning kNN
Entrainement et test sur les données statiques
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
filename = 'Micro_1s_sample_0s_ti'  # fichier conteant les features d'apprentissage
ti = 0.1    # durée de recouvrement en secondes (pas entre chaque échantillon)
#############################################################################

t0 = datetime.now() 

# importation des features et des labels

# read pickle file
infile = open('Datas\\Pickle\\Features\\Features_' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

X = df[0]   # matrice X contenant les features
y = df[1]   # vecteur y conteant les lables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 70% training and 30% test, random_state pour avoir toujours les mêmes données en train,
# stratify = y pour avoir le même poucentage de label que les sets

kmax = int(np.sqrt(len(y)))  # nombre de voisin maximal à tester 
# création de dictionnaire pour quantifier l'erreur des différents types de mesures
euclid = {'k': [], 'errortrain': [], 'errortest': []}   # mesure la plus direct entre deux points
manathan = {'k': [], 'errortrain': [], 'errortest': []}     # mesure par quadrillage (distance la plus longue)

# création du modèle kNN
knn = KNeighborsClassifier(n_neighbors=int(kmax)) # p = 1 for manhattan distance
print('Facteur k = ' + str(kmax) + ' et distance euclidienne')

# Train the model using the training sets
knn.fit(X_train, y_train)
errortrain = 1 - knn.score(X_train, y_train)  # calcul de l'erreur d'entrainement
y_pred = knn.predict(X_test) # test de prédiction
errortest = 1 - knn.score(X_test, y_test) # calcul de l'erreur de test

# affichage des résultats
print('Erreur train: %f' % errortrain)
print('Erreur test: %f' % errortest)
print("Accuracy:" + str(metrics.accuracy_score(y_test, y_pred)))

# boucle  pour tester l'erreur en fonction du nombre de voisin 2 à kmax
# et du type de mesure (euclidienne ou manhattan)
errorstrain = []
errorstest = []
print('\nMesure distance Euclidienne')
for k in range(2, kmax):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, p=2)
    errorstrain.append(1 - knn.fit(X_train, y_train).score(X_train, y_train))
    errorstest.append(1 - knn.fit(X_train, y_train).score(X_test, y_test))
indexmin = errorstest.index(min(errorstest)) + 2
Errormintrain = min(errorstrain)
Errormintest = min(errorstest)
plt.plot(range(2, kmax), errorstest, 'o-')
plt.xlabel('Number of neighbors')
plt.ylabel('Error')
plt.show()
# choix du nombre de voisin optimal
knn = neighbors.KNeighborsClassifier(indexmin)
print('Facteur k choisi : ' + str(indexmin))
print('Min erreur train: ' + str(min(errorstrain)))
print('Erreur test avec k choisi: ' + str(min(errorstest)))

knnPickle = open('Datas\\Pickle\\kNN\\kNN_' + filename + '.pckl', 'wb') 

# source, destination 
pickle.dump(knn, knnPickle) 

#affichage du temps écoulé
t1 = datetime.now()
print('\nTemps écoulé : ' + str(t1-t0) + ' [h:m:s]')