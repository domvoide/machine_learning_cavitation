# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:29:06 2020

@author: voide

plot de toutes les variances
"""
import pickle
import statistics
from matplotlib import pyplot as plt

filename = 'Short_1s_sample_0.9s_step'
# read pickle file
infile = open('Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

fig = plt.figure(figsize=(10,6))

def plotVar(x, audio,color):
    var = statistics.variance(audio)
    plt.scatter(x, var, color=color)

for i in range(len(df.Micro)):
    if df.Cavit[i] == 0:
        c = 'b'
        x = 0
    else:
        c = 'r'
        x = 1
    plotVar(x, df.Micro[i], c)