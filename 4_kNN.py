# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:12:24 2020

@author: voide
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics
from sklearn import neighbors
from matplotlib import pyplot as plt

# importation des features et des labels
filename = 'Short_1s_sample_0.9s_step'
# read pickle file
infile = open('Datas\\Pickle\\Features\\Features' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

X = df[0]
y = df[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

kmax = int(np.sqrt(len(y)))
euclid = {'k': [], 'errortrain': [], 'errortest': []}
manathan = {'k': [], 'errortrain': [], 'errortest': []}

# création du modèle kNN
knn = KNeighborsClassifier(n_neighbors=int(np.sqrt(len(y)))) # p = 1 for manhattan distance
print('Facteur k = ' + str(kmax) + ' et distance euclidienne')
# Train the model using the training sets
knn.fit(X_train, y_train)
errortrain = 1 - knn.score(X_train, y_train)
y_pred = knn.predict(X_test)
errortest = 1 - knn.score(X_test, y_test)
print('Erreur train: %f' % errortrain)
print('Erreur test: %f' % errortest)
print("Accuracy:" + str(metrics.accuracy_score(y_test, y_pred)))


for i in range(10):
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
    # plt.plot(range(2, kmax), errorstest, 'o-')
    # plt.xlabel('Number of neighbors')
    # plt.ylabel('Error')
    # plt.show()
    
    knn = neighbors.KNeighborsClassifier(indexmin)
    print('Facteur k choisi : ' + str(indexmin))
    print('Min erreur train: ' + str(min(errorstrain)))
    print('Erreur test avec k choisi: ' + str(min(errorstest)))
    euclid['k'].append(indexmin)
    euclid['errortrain'].append(Errormintrain)
    euclid['errortest'].append(Errormintest)
    
    print('\nMesure distance manhattan')
    errorstrain = []
    errorstest = []
    for k in range(2, kmax):
        knn = neighbors.KNeighborsClassifier(n_neighbors=k, p=1)
        errorstrain.append(1 - knn.fit(X_train, y_train).score(X_train, y_train))
        errorstest.append(1 - knn.fit(X_train, y_train).score(X_test, y_test))
    indexmin = errorstest.index(min(errorstest)) + 2
    Errormintrain = min(errorstrain)
    Errormintest = min(errorstest)
    # plt.plot(range(2, kmax), errorstest, 'o-')
    # plt.xlabel('Number of neighbors')
    # plt.ylabel('Error')
    # plt.show()
    
    knn = neighbors.KNeighborsClassifier(indexmin)
    print('Facteur k choisi : ' + str(indexmin))
    print('Min erreur train: ' + str(min(errorstrain)))
    print('Erreur test avec k choisi: ' + str(min(errorstest)))
    manathan['k'].append(indexmin)
    manathan['errortrain'].append(Errormintrain)
    manathan['errortest'].append(Errormintest)
