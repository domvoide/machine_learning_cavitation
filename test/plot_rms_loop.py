# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:29:06 2020

@author: voide

plot de toutes les variances et de la moyenne de chaque label
"""

import numpy as np
import pickle
import statistics
from matplotlib import pyplot as plt
from datetime import datetime

t0 = datetime.now()
filename = '1s_sample_0.1s_ti'
# read pickle file

infile = open('..\\Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

cavit = []
no_cavit = []

fig = plt.figure(figsize=(5,3))  # création figure
ax = fig.add_subplot(211)
# Boucle pour plotter toutes les variances
for i in range(len(df.Micro)):
    if df.Cavit[i] == 0:
        c = 'b'
        y = 0
        rms = np.sqrt(sum(n*n for n in df.Micro[i])/len(df.Micro[i]))
        ax.scatter(rms, y, color=c)
        no_cavit.append(rms)
    else:
        c = 'r'
        y = 1
        rms = np.sqrt(sum(n*n for n in df.Micro[i])/len(df.Micro[i]))
        ax.scatter(rms, y, color=c)
        cavit.append(rms)
    
# Mise en forme du graphique
ax.set_ylim(-0.5,1.5)
ax.set_yticks((0, 1))
ax.set_yticklabels(('non', 'oui'), rotation='vertical')
ax.set_xlabel('Root mean square [-]')
ax.set_ylabel('Cavitation')


# moyenne des variances par label
mean_no_cavit = np.mean(no_cavit)
mean_cavit = np.mean(cavit)

# plot des moyennes
plt.scatter(mean_no_cavit, 0, c='black', edgecolor='white', marker='x', s=100)
plt.scatter(mean_cavit, 1, c='black', edgecolor='white', marker='x', s=100)

#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0))
