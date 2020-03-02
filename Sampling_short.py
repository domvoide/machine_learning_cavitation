import io
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import sounddevice as sd

##############################################################################
# Paramètres
dt = 2  # durée de l'échantillon en secondes
ti = 1.8  # durée de recouvrement en secondes (pas entre chaque échantillon)
##############################################################################


# importation du pickle contenant les 18 fichiers de données
folder_path = 'all_data.pckl'
buf = io.open(folder_path, 'rb', buffering=(128 * 2048))
dataframe = pickle.load(buf)
buf.close()
del buf
dataframe = dataframe.transpose()


# fonction pour vérifier l'utilisation de la mémoire
def memory_usage(df):
    return round(df.memory_usage(deep=True) / 1024**2, 2)


# creation d'une matrice plus compact pour commencer (plus rapide à travailler)
df2 = dataframe[['Alpha', 'Sigma', 'Time', 'Micro', 'Cavit']].copy()
del dataframe

# Acquisition de la fréquence
f = round(1 / (df2.Time[0][1]-df2.Time[0][0]), 1)


# Création d'un dictionnaire vide pour les échantillons d'analyses
data = {'Alpha': [],
        'Sigma': [],
        'Micro': [],
        'Cavit': []}


size = int(dt * f)  # longueur de l'échantillon
step = int(ti * f)  # écart entre le début de chaque échantillon

# création des échantillons
for j in range(18):
    for i in range(0, len(df2.Micro[j]), step):
        a = df2.Micro[j][i: i + size]
        a = a.astype('float64')
        data['Micro'].append(a)
        data['Cavit'].append(df2.Cavit[j])
        data['Alpha'].append(df2.Alpha[j])
        data['Sigma'].append(df2.Sigma[j])

# transformation en dataframe et mise en forme pour compacter
df = pd.DataFrame(data)
df.Cavit = df.Cavit.astype('uint8')
df.Alpha = df.Alpha.astype('float32')
df.Sigma = df.Sigma.astype('float32')

print('Memory used:', memory_usage(df), 'Mb')

# Affichage du graphe des points de fonctionnements testés
fig, ax = plt.subplots(figsize=(11, 8))
scatter = ax.scatter(df2.Alpha, df2.Sigma, c=df2.Cavit, cmap='Accent')
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Cavitation")
ax.add_artist(legend1)
plt.xlabel(r'$\alpha$ [°]')
plt.ylabel(r'$\sigma$ [-]')
plt.grid()
plt.show()

f = open('Datas\\Short_' + str(dt) + 's_sample_' + str(ti) +
         's_step.pckl', 'wb')
pickle.dump(df, f)
f.close()


def sound(track):
    sd.play(track, 44100)
