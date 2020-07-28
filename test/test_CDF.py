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

infile = open('Datas\\Pickle\\Sampling\\Short_1s_sample_0.9s_step.pckl', 'rb')
df = pickle.load(infile)
infile.close()

X = df.Micro

norm_cdf_no_cavit = np.sort(X[0])
norm_cdf_cavit = np.sort(X[322])

p_no_cavit = np.linspace(0, 1, len(X[0]))
p_cavit = np.linspace(0, 1, len(X[322]))

# plot the sorted data:
fig = plt.figure(figsize=(8, 6))
fig.suptitle('CDF with numpy sort', y=1.02, fontsize=15)

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
ax2.set_ylim(-3, 3)
ax2.label_outer()

print('Numpy Order')
print('CDF pas de cavitation : ' + str(max(norm_cdf_no_cavit)))
print('CDF cavitation : ' + str(max(norm_cdf_cavit)))

norm_cdf_no_cavit2 = scipy.stats.norm.cdf(X[0])
norm_cdf_cavit2 = scipy.stats.norm.cdf(X[322])

fig = plt.figure(figsize=(8, 6))
fig.suptitle('CDF with scipy stats norm', y=1.02, fontsize=15)

ax1 = fig.add_subplot(121)
sns.lineplot(x=norm_cdf_no_cavit2, y=X[0],  ci=None)
plt.title('No Cavitation')


ax1.set_xlabel('$p$')
ax1.set_ylabel('$x$')
ax1.set_ylim(-3, 3)
ax1.set_xlim(0, 1)

ax2 = fig.add_subplot(122)
sns.lineplot(x=norm_cdf_cavit2, y=X[322], ci=None)
plt.title('Cavitation')
ax2.set_xlabel('$p$')
ax2.set_ylabel('$x$')
ax2.set_ylim(-3, 3)

ax2.label_outer()

plt.show()
print('Scipy stats norm CDF')
print('CDF pas de cavitation : ' + str(max(X[0])))
print('CDF cavitation : ' + str(max(X[322])))
