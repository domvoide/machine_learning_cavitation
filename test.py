import pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal



infile = open('Datas\\Pickle\\Sampling\\Short_1s_sample_0.9s_step.pckl', 'rb')
df = pickle.load(infile)
infile.close()

y = df.Micro[40]
rate = 40000
dt = 1/rate

L = len(y)
NFFT = int(2 ** np.ceil(np.log2(abs(L))))
Fs = rate
freq = Fs / 2 * np.linspace(0,1,NFFT//2)

print(f'Audio length: {L:.2f} seconds')

freqs, psd = signal.welch(y, fs=rate)

plt.figure(figsize=(5, 4))
plt.semilogx(freqs, psd)
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()
maxPSD = max(psd)
print(maxPSD)