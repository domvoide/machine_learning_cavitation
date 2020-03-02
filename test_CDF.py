# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:15:20 2020

@author: dominiqu.voide
"""
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

infile = open('Datas\\Short_2s_sample_1.8s_step.pckl', 'rb')
df = pickle.load(infile)
infile.close()

X = df.Micro

norm_cdf_no_cavit = scipy.stats.norm.cdf(X[0])
norm_cdf_cavit = scipy.stats.norm.cdf(X[148])

# plot the cdf
sns.lineplot(x=X[0], y=norm_cdf_no_cavit)
plt.show()

sns.lineplot(x=X[148], y=norm_cdf_cavit)
plt.show()