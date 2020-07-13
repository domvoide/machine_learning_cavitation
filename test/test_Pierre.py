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
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=1)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise']
lw = 2

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of data')
plt.xlabel('1st PCA dimension')
plt.ylabel('2nd PCA dimension')

plt.figure()
for color, i, target_name, tag in zip(colors, [0, 1], target_names,{'x','o'}):
    plt.plot(numpy.where(y==i)[0],X_r2[y == i, 0], alpha=.8, color=color,
                label=target_name, marker=tag, linestyle='None')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of data')
plt.xlabel('Sample number')
plt.ylabel('LDA value')

plt.show()