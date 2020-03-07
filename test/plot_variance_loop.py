# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:29:06 2020

@author: voide

plot de toutes les variances
"""
import os
import pickle
import statistics
from matplotlib import pyplot as plt
from datetime import datetime

t0 = datetime.now()
filename = 'Short_1s_sample_0.9s_step'
# read pickle file

infile = open('..\\Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()


fig = plt.figure(figsize=(10,6))  # création figure

def plotVar(audio, y, color):
    '''
    Fonction pour plotter un point pour la variance d'un signal'
    Parameters
    ----------
    audio : array
        Signal audio à analyser.
    y : int
        position en y du plot 1D.
    color : str
        couleur du point en fonction du label.

    Returns
    -------
    Scatter de la variance

    '''
    var = statistics.variance(audio)
    plt.scatter(var, y, color=color)

# Boucle pour plotter toutes les variances
for i in range(len(df.Micro)):
    if df.Cavit[i] == 0:
        c = 'b'
        y = 0
    else:
        c = 'r'
        y = 1
    plotVar(df.Micro[i], y, c)
    
# Mise en forme du graphique
plt.text(1.6, 1, 'Red : cavitation\nBlue : no cavitation', fontsize = 15)
plt.xlabel('Variance [-]')
plt.ylabel('Cavitation [-]')

t1 = datetime.now()
print('Temps écoulé : ' + (t1-t0))