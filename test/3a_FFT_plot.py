import pickle
from matplotlib import pyplot as plt
import numpy as np




infile = open('Datas\\Pickle\\Sampling\\Short_1s_sample_0.9s_step.pckl', 'rb')
df = pickle.load(infile)
infile.close()

audio = df.Micro[1060]
f = 40000

N = audio.shape[0]
L = N / f

print(f'Audio length: {L:.2f} seconds')

fig, ax = plt.subplots()
ax.plot(np.arange(N) // f, audio)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude [unknown]');

yf = np.abs(np.fft.fft(audio))
fftmax = np.abs(np.sort(-yf[0:int(N/2)])[:10])
fig1, ax1 = plt.subplots()
ax1.semilogx(np.arange(N//2) / f, yf[0:int(N/2)])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Fréquence [Hz]')
print(fftmax)
