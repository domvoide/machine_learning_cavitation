# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:36:44 2020

@author: dominiqu.voide
"""

from sklearn import svm
import pickle

# read pickle file
infile = open('Data0.pckl', 'rb')
df= pickle.load(infile)
infile.close()

X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
