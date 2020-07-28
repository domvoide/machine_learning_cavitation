# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:12:24 2020

@author: voide

Algorithme de machine learning kNN
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt
from datetime import datetime

# Paramètres
#############################################################################
filename = 'Features_Micro_1s_sample_0s_ti'  # fichier conteant les features d'apprentissage
ti = 0.1    # durée de recouvrement en secondes (pas entre chaque échantillon)
#############################################################################

t0 = datetime.now() 

# importation des features et des labels

# read pickle file
infile = open('Datas\\Pickle\\Features\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

X = df[0]   # matrice X contenant les features
y = df[1]   # vecteur y conteant les lables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
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

print('Pourcentage d\'erreur : ' + str(errortest))

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
X_New = clf.fit_transform(X,y)

knn_new = KNeighborsClassifier(n_neighbors=3)
knn_new.fit(X,y)
errortrain_new = 1 - knn_new.score(X, y)

#affichage du temps écoulé
t1 = datetime.now()
print('\nTemps écoulé : ' + str(t1-t0) + ' [h:m:s]')