import pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

data = np.empty(0)

a = 1
b = 2
c = 3

for i in range(5):
    data = np.append(data,a)
    data = np.append(data,b)
    data = np.append(data,c)

data = np.reshape(data,(3,5))
