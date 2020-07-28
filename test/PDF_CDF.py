# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:25:18 2020

@author: voide
"""
from datetime import datetime
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import mlab
import statistics
import scipy
from scipy import signal
from scipy import stats
from scipy.stats import norm

t0 = datetime.now()
filename = '1s_sample_0.1s_ti'
# read pickle file
infile = open('..\\Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()


audio = df.Micro[85]  # fichier audio brut
Fs = 40000  # fréquence d'acquisition

# var = statistics.variance(audio)

L = len(audio)
NFFT = int(2 ** np.ceil(np.log2(abs(L))))
# freq = Fs / 2 * np.linspace(0, 1, NFFT//2)

T = 1.0 / Fs

########################################################################
# fft
xf = np.linspace(0.0, 1.0/(2.0*T), NFFT//2)
yf = np.fft.fft(audio, NFFT)/L
yf = 2*abs(yf[:NFFT//2])

# on néglige les x premières fréquences
xf = xf[20:]
yf = yf[20:]

fig = plt.figure(figsize=(10,15))
fig.add_subplot(311)
plt.title('FFT')
plt.plot(xf, yf)

########################################################################
# PDF

mean_yf = np.mean(audio)
std_yf = np.std(audio)
dist = norm(mean_yf, std_yf)
values = [value for value in range(0, 20000, 10)]
prob = [dist.pdf(value) for value in values]
fig.add_subplot(312)
plt.title('PDF')
plt.plot(values, prob)


########################################################################
# CDF

cumsum = np.cumsum(yf)
fig.add_subplot(313)
plt.title('CDF')
plt.plot(xf, cumsum/sum(yf))


# generate a sample
sample = np.random.normal(loc=50, scale=5, size=1000)
# calculate parameters
sample_mean = np.mean(sample)
sample_std = np.std(sample)
print('Mean=%.3f, Standard Deviation=%.3f' % (sample_mean, sample_std))
# define the distribution
dist = norm(sample_mean, sample_std)
# sample probabilities for a range of outcomes
values = [value for value in range(30, 70)]
probabilities = [dist.pdf(value) for value in values]
# plot the histogram and pdf
fig = plt.figure()
ax = fig.add_subplot(211)
ax.hist(sample, bins=10, density=True)
ax.plot(values, probabilities)

x = np.linspace(30, 70, 1000) 
ax2 = fig.add_subplot(212)
ax2.plot(x, sample, c='red')




