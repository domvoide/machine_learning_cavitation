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

fig = plt.figure(figsize=(10,6))  # création figure

# Boucle pour plotter toutes les variances
for i in range(len(df.Micro)):
    if df.Cavit[i] == 0:
        c = 'b'
        y = 0
        var = statistics.variance(df.Micro[i])
        plt.scatter(var, y, color=c)
        no_cavit.append(var)
    else:
        c = 'r'
        y = 1
        var = statistics.variance(df.Micro[i])
        plt.scatter(var, y, color=c)
        cavit.append(var)
    
# Mise en forme du graphique
plt.text(1.6, 0.8, 'Red : cavitation\nBlue : no cavitation\nx : mean', fontsize = 15)
plt.xlabel('Variance [-]')
plt.ylabel('Cavitation [-]')

# moyenne des variances par label
mean_no_cavit = np.mean(no_cavit)
mean_cavit = np.mean(cavit)

# plot des moyennes
plt.scatter(mean_no_cavit, 0, c='black', edgecolor='white', marker='x', s=100)
plt.scatter(mean_cavit, 1, c='black', edgecolor='white', marker='x', s=100)

#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0))
