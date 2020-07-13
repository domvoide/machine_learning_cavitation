import pickle
from matplotlib import pyplot as plt
import numpy as np


n = 0

infile = open('..\\Datas\\Pickle\\Sampling\\1s_sample_0s_ti.pckl', 'rb')
df = pickle.load(infile)
infile.close()

f = 40000
N = len(df.Micro[0])
L = N / f
print(f'Audio length: ' + str(L) + ' seconds')
fftmax = []
label = []
label_txt = []
color = []

def fftfreq(audio, Fs):
    L = len(audio)
    NFFT = int(2 ** np.ceil(np.log2(abs(L))))
    freq = Fs / 2 * np.linspace(0, 1, NFFT//2)

    yf = np.fft.fft(audio, NFFT)/L
    amp = 2 * abs(yf[1:NFFT//2+1])
    
    # Déterminer les 10 peaks les plus importants
    freqmax_index = abs(np.argsort(-amp[100:NFFT//2])[:1]) # néglige les fréquences en dessous de 100 Hz
    freqmax = freq[freqmax_index]
    return freqmax

for n in range(len(df.Micro)):
    audio = df.Micro[n]
    label_audio = df.Cavit[n]
    if label_audio == 0:
        label_audio_txt = 'Non Cavitant'
        c = 'blue'
    else:
        label_audio_txt = 'Cavitant'
        c = 'red'
    label_txt.append(label_audio_txt)
    color.append(c)
    label.append(label_audio)
    fftmax.append(fftfreq(audio, f))
    
    
    
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
ax.scatter(fftmax,label, color=color, label=label_txt)
ax.set_ylim(-0.5,1.5)
ax.set_yticks((0, 1))
ax.set_yticklabels(('non', 'oui'), rotation='vertical')
ax.set_xlabel('Fréquence maximale FFT [Hz]')
ax.set_ylabel('Cavitation')
