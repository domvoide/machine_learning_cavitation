# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:44:07 2020

@author: voide

Affichage des graphes des mesures de centrale
CDF et PCA avec la mesure statique pour comparaison
enregistrement au format png sous analyse_feature

"""

import matplotlib.pyplot as plt
import pickle
import numpy as np


analyse_type = 'dynamique' # statique ou dynamique  ou centrale
if analyse_type == 'statique':
    lda_file = 'lda_Features_Micro_1s_sample_0s_ti'
    pca_file = 'pca_Features_Micro_1s_sample_0s_ti'
else:
    lda_file = 'lda_Features_Micro_Dyn_01_1s_sample_0s_ti'
    pca_file = 'pca_Features_Micro_Dyn_01_1s_sample_0s_ti'

filename = 'Features_Centrale_statique'  # fichier conteant les features d'apprentissage
# importation des features et des labels
# read pickle file

infile = open('Datas\\Pickle\\Features\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

X = df   # matrice X contenant les features

# pca = PCA(n_components=2)
# X_r = pca.fit(X)

pca = pickle.load(open('Datas\\Pickle\\kNN\\' + pca_file + '.pckl', 'rb'))
X_r = pca.transform(X)

lda = pickle.load(open('Datas\\Pickle\\kNN\\' + lda_file + '.pckl', 'rb'))
X_r2 = lda.transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
lw = 2

plt.scatter(X_r[:,0], X_r[:,1], alpha=.8, lw=lw)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Annalyse PCA ' + analyse_type +' de la Centrale')
plt.xlabel('1st PCA dimension')
plt.ylabel('2nd PCA dimension')

plt.figure()
x = np.arange(0, len(X_r2), 1)
plt.scatter(x, X_r2[:], alpha=.8)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Annalyse LDA ' + analyse_type +' de la Centrale')
plt.xlabel('Sample number')
plt.ylabel('LDA value')