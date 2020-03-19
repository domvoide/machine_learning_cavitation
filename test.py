import pickle
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import numpy as np
from scipy import signal
import os

filename = '1s_sample_0.1s_ti'
infile = open('Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

fig = plt.figure(1, figsize=(20, 6.5))
ax = fig.add_subplot(121)
color=plt.cm.tab10(np.linspace(0,1,9))
lenptfct = int(len(df.Alpha)/18)
for i,c in zip(range(0, int(len(df.Alpha)/2), lenptfct), color):
    ax.scatter(df.Alpha[i:i+lenptfct], df.Sigma[i:i+lenptfct],c=c, marker='x',
               label= 'Pt : ' + str(int((i+lenptfct)/lenptfct)))
for i,c in zip(range(int(len(df.Alpha)/2), len(df.Alpha), lenptfct), color):
    ax.scatter(df.Alpha[i:i+lenptfct], df.Sigma[i:i+lenptfct],edgecolors=c, marker='o',
               facecolors='none', label= 'Pt : ' + str(int((i+lenptfct)/lenptfct)))

ax.legend(loc='best', ncol=2, )
ax.set_xlabel(r'$\alpha$ [°]')
ax.set_ylabel(r'$\sigma$ [-]')
ax.grid()


ax2 = fig.add_subplot(122)
scatter = ax2.scatter(df.Alpha, df.Sigma, c=df.Cavit, cmap='bwr')
legend1 = ax2.legend(*scatter.legend_elements(),
                    loc="upper left", title="Cavitation")
ax2.add_artist(legend1)
ax2.set_xlabel(r'$\alpha$ [°]')
ax2.set_ylabel(r'$\sigma$ [-]')
ax2.grid()

plt.tight_layout()