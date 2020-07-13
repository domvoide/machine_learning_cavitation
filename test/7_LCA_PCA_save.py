print(__doc__)

import matplotlib.pyplot as plt
import pickle
import numpy
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

filename = 'Features_Micro_Dyn_01_1s_sample_0s_ti'  # fichier conteant les features d'apprentissage
# importation des features et des labels
# read pickle file

infile = open('Datas\\Pickle\\Features\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

X = df[0]   # matrice X contenant les features
y = df[1]   # vecteur y conteant les lables
target_names = {'0','1'}

pca = PCA(n_components=2)
X_r = pca.fit(X)

lda = LinearDiscriminantAnalysis(n_components=1)
X_r2 = lda.fit(X, y)

pcaPickle = open('Datas\\Pickle\\kNN\\pca_' + filename + '.pckl', 'wb') 

# source, destination 
pickle.dump(X_r, pcaPickle) 

ldaPickle = open('Datas\\Pickle\\kNN\\lda_' + filename + '.pckl', 'wb') 

# source, destination 
pickle.dump(X_r2, ldaPickle) 