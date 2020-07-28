import pickle
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import numpy as np
from scipy import signal
import os


datatype = 'Acc_Z' # choix du capteur 'Acc_X', 'Acc_Y', 'Acc_Z', 'Micro', 'Uni_Y', 'Hydro'
filename = '1s_sample_0.1s_ti'
fCDF = np.arange(0, 21000, 1000)
Fs = 40000  # fréquence d'acquisition
fftstart = 20  # valeurs de fréquence à négliger au début
pathfig = 'C:\\Users\\voide\\Documents\\Master\\Machine Learning\
\Documents\\Analyse_features\\'

# read pickle file
infile = open('Datas\\Pickle\\Sampling\\' + filename + '.pckl', 'rb')
df = pickle.load(infile)
infile.close()

# sets de couleurs pour meilleure visualisation des points
color = plt.cm.tab20(np.linspace(0,1,18))

fig2 = plt.figure(figsize=(20,13))
ax4 = fig2.add_subplot(224)
lenptfct = int(len(df.Alpha)/18)
for i,c in zip(range(0, len(df.Alpha), lenptfct), color):
    if df.Cavit[i] == 0:
        mkr = 'x'
        face = c
    else: 
        mkr = 'o'
        face ='none'

    ax4.scatter(df.Alpha[i:i+lenptfct], df.Sigma[i:i+lenptfct],edgecolors=c, 
               marker=mkr, s=75,
               facecolors=face, label= 'Pt : ' + str(int((i+lenptfct)/lenptfct)))

a = np.arange(1 , 16, 1)
b = 0.003 * a ** 3 - 0.035*a**2 + 0.6231*a + 0.8694
ac = np.arange(7, 10.3, 0.1)
c = 0.125 * ac**2 -2.975 * ac + 19.225
d = 0.0833 * a + 1
ax4.plot(a, b, ac, c, a, d, c='k')

ax4.text(6, 6.5,'No Cavitation', fontsize=14, 
        bbox=dict(boxstyle='round', alpha=0.5, facecolor='white'))
ax4.text(2.75, 1.65,'Bubble Cavitation', fontsize=14, rotation=10,
        bbox=dict(boxstyle='round', alpha=0.5, facecolor='white'))
ax4.text(11, 5.5,'Pocket Cavitation', fontsize=14,
        bbox=dict(boxstyle='round', alpha=0.5, facecolor='white'))
ax4.text(7, 0.85,'Supercavitation', fontsize=14, rotation=3,
        bbox=dict(boxstyle='round', alpha=0.5, facecolor='white'))

ax4.legend(loc='best', ncol=2, )
ax4.set_xlabel(r'$\alpha$ [°]')
ax4.set_ylabel(r'$\sigma$ [-]')
ax4.grid()