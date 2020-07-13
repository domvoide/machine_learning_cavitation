# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:12:24 2020

@author: voide

Algorithme de machine learning kNN
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Paramètres
#############################################################################
filename = 'Features_Micro_1s_sample_0s_ti'  # fichier conteant les features d'apprentissage

n_neighbors = 5
random_state = 0
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

dim = len(X[0])
n_classes = len(np.unique(y))

# Reduce dimension to 2 with PCA
pca = make_pipeline(StandardScaler(),
                    PCA(n_components=2, random_state=random_state))

# Reduce dimension to 2 with LinearDiscriminantAnalysis
lda = make_pipeline(StandardScaler(),
                    LinearDiscriminantAnalysis(n_components=2))

# Reduce dimension to 2 with NeighborhoodComponentAnalysis
nca = make_pipeline(StandardScaler(),
                    NeighborhoodComponentsAnalysis(n_components=2,
                                                   random_state=random_state))

# Use a nearest neighbor classifier to evaluate the methods
knn = KNeighborsClassifier(n_neighbors=n_neighbors)

# Make a list of the methods to be compared
dim_reduction_methods = [('PCA', pca), ('LDA', lda), ('NCA', nca)]

# plt.figure()
for i, (name, model) in enumerate(dim_reduction_methods):
    plt.figure()
    # plt.subplot(1, 3, i + 1, aspect=1)

    # Fit the method's model
    model.fit(X_train, y_train)

    # Fit a nearest neighbor classifier on the embedded training set
    knn.fit(model.transform(X_train), y_train)

    # Compute the nearest neighbor accuracy on the embedded test set
    acc_knn = knn.score(model.transform(X_test), y_test)

    # Embed the data set in 2 dimensions using the fitted model
    X_embedded = model.transform(X)

    # Plot the projected points and show the evaluation score
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap='Set1')
    plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name,
                                                              n_neighbors,
                                                              acc_knn))
plt.show()
#affichage du temps écoulé
t1 = datetime.now()
print('\nTemps écoulé : ' + str(t1-t0) + ' [h:m:s]')