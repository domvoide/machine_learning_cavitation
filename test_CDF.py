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

norm_cdf_no_cavit = np.sort(X[0])
norm_cdf_cavit = np.sort(X[148])

p_no_cavit = 1. * np.arange(len(X[0])) / (len(X[0]) - 1)
p_cavit = 1. * np.arange(len(X[148])) / (len(X[148]) - 1)

# plot the sorted data:
fig = plt.figure()
fig.suptitle('Cumulative distribution function, CDF', y=1.02, fontsize=15)

ax1 = fig.add_subplot(121)
ax1.plot(p_no_cavit, norm_cdf_no_cavit)
plt.title('No Cavitation')
ax1.set_xlabel('$p$')
ax1.set_ylabel('$x$')
ax1.set_ylim(-3, 3)

ax2 = fig.add_subplot(122)
ax2.plot(p_cavit, norm_cdf_cavit)
plt.title('Cavitation')
ax2.set_xlabel('$p$')
ax2.set_ylabel('$x$')
ax2.label_outer()


print('CDF pas de cavitation : '+ str(max(norm_cdf_no_cavit)))
print('CDF cavitation : ' + str(max(norm_cdf_cavit)))