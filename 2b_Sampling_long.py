import io
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import sounddevice as sd


# importation du pickle contenant les 18 fichiers de données
folder_path = 'C:\\Users\\voide\\Documents\\Master\\Machine Learning\\Datas\
\\Pickle\\all_data.pckl'
buf = io.open(folder_path, 'rb', buffering=(128 * 2048))
df1 = pickle.load(buf)
buf.close()
del buf
df1 = df1.transpose()


# fonction pour vérifier l'utilisation de la mémoire
def memory_usage(df):
    return(round(df.memory_usage(deep=True) / 1024, 2))


# Acquisition de la fréquence
f = round(1 / (df1.Time[0][1]-df1.Time[0][0]), 1)


# Création d'un dictionnaire vide pour les échantillons d'analyses
data = {'Alpha': [],
        'Sigma': [],
        'Acc_X': [],
        'Acc_Y': [],
        'Acc_Z': [],
        'Micro': [],
        'Uni': [],
        'Hydro': [],
        'AE': [],
        'Cavit': []}

dt = 5  # durée de l'échantillon en secondes
ti = 2  # durée de recouvrement en secondes (pas entre chaque échantillon)

size = int(dt * f)  # longueur de l'échantillon
step = int(ti * f)  # écart entre le début de chaque échantillon

# création des échantillons
for j in range(18):
    for i in range(0, len(df1.Micro[j]), step):
        a = df1.Micro[j][i: i + size]
        # a = a.astype('float64')
        data['Acc_X'].append(df1.Acc_X[j][i: i + size])
        data['Acc_Y'].append(df1.Acc_Y[j][i: i + size])
        data['Acc_Z'].append(df1.Acc_Z[j][i: i + size])
        data['Micro'].append(df1.Micro[j][i: i + size])
        data['Uni'].append(df1.Uni[j][i: i + size])
        data['Hydro'].append(df1.Hydro[j][i: i + size])
        data['AE'].append(df1.Brut[j][i: i + size])
        data['Cavit'].append(df1.Cavit[j])
        data['Alpha'].append(df1.Alpha[j])
        data['Sigma'].append(df1.Sigma[j])

# transformation en dataframe et mise en forme pour compacter
df = pd.DataFrame(data)
df.Cavit = df.Cavit.astype('uint8')
df.Alpha = df.Alpha.astype('float32')
df.Sigma = df.Sigma.astype('float32')

print('Memory used:', memory_usage(df), 'Mb')

# Affichage du graphe des points de fonctionnements testés
fig, ax = plt.subplots(figsize=(11, 8))
scatter = ax.scatter(df1.Alpha, df1.Sigma, c=df1.Cavit, cmap='Accent')
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Cavitation")
ax.add_artist(legend1)
plt.xlabel(r'$\alpha$ [°]')
plt.ylabel(r'$\sigma$ [-]')
plt.grid()
plt.show()

f = open('Datas\\Pickle\\Sampling\\Long_' + str(dt) + 's_sample_' + str(ti) +
         's_step.pckl', 'wb')
pickle.dump(df, f)
f.close()


def sound(track):
    sd.play(track, 44100)
