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

#importation des features et des labels
filename = 'Short_1s_sample_0.9s_step'
# read pickle file
infile = open('Datas\\Pickle\\Features\\Features' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

X = df[0]
y = df[1]

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X2 = sel.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3) # 70% training and 30% test


# création du modèle kNN
knn = KNeighborsClassifier(n_neighbors=6, p=1)

# Train the model using the training sets
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
error = 1 - knn.score(X_test, y_test)
print('Erreur: %f' % error)
print("Accuracy:" + str(metrics.accuracy_score(y_test, y_pred)))

errors = []
for k in range(2,200):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(1 - knn.fit(X_train, y_train).score(X_test, y_test))
indexmin = errors.index(min(errors)) + 2
plt.plot(range(2,200), errors, 'o-')
plt.xlabel('Number of neighbors')
plt.ylabel('Error')
plt.show()

knn = neighbors.KNeighborsClassifier(indexmin)
print('Facteur k choisi : ' + str(indexmin))
print('Erreur : ' + str(min(errors)))
knn.fit(X_train, y_train)


print("Accuracy:" + str(metrics.accuracy_score(y_test, y_pred)))
