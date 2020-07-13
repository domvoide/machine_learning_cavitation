import pickle
from matplotlib import pyplot as plt
import numpy as np


n = 1000

infile = open('..\\Datas\\Pickle\\Sampling\\1s_sample_0s_ti.pckl', 'rb')
df = pickle.load(infile)
infile.close()

audio = df.Micro[n]
label_audio = df.Cavit[n]
if label_audio == 0:
    label_audio = 'Non'
else:
    label_audio = 'Oui'
f = 40000

N = audio.shape[0]
L = N / f

print(f'Audio length: {L:.2f} seconds')

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
ax.plot(np.arange(N) // f, audio)

ax.set_ylabel('Amplitude [-]');

yf = np.abs(np.fft.fft(audio))
fftmax = np.abs(np.sort(-yf[0:int(N/2)])[:10])

ax1 = fig.add_subplot(212)
ax1.semilogx(np.arange(N//2) / f, yf[0:int(N/2)])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Fréquence [Hz]')
print('fft max :')
print(fftmax)
print('Cavitation : ' + str(label_audio))
