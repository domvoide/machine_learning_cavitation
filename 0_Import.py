
"""
Created on Wed Feb 07 10:12:35 2020

@author: dominiqu.voide

Importation des différents fichiers tdms fournis par François Pedroni
Il y a 3 fichiers par point de fonctionnement qu'il faut renseigner en path 1,2,3
il faut également renseigner le sigma et alpha du point et le label de cavitaiton

Si nécessaire installer le package nptdms via commande --> pip install nptdms dans le terminal
"""

from nptdms import TdmsFile
import pickle
import sounddevice as sd
import numpy as np
import pandas as pd

# Paramètres à modifier
#############################################################################
cavitation = 0  # 0 pour non et 1 pour oui
alpha = 6
sigma = 4.4
# Mettre le chemin d'accès des 3 fichiers de la même mesure
path1 = 'Datas\\No_cavitation\\Sigma4p4\\Mesures_Turbicav_AE_RMS_pressure_strain_geo_alpha6_40kHz_19-12-04_1044_001.tdms'
path2 = 'Datas\\No_cavitation\\Sigma4p4\\Mesures_Turbicav_AE_RMS_pressure_strain_geo_alpha6_40kHz_19-12-04_1044_002.tdms'
path3 = 'Datas\\No_cavitation\\Sigma4p4\\Mesures_Turbicav_AE_RMS_pressure_strain_geo_alpha6_40kHz_19-12-04_1045_003.tdms'
#############################################################################

Name = 'Alpha_' + str(alpha) + '_Sigma_' + str(sigma) # nom du fichier pour enregistrement pickle
paths = [path1, path2, path3]
# création des dictionnaire pour chaque vecteur
properties, names, time,  acc_X, acc_Y, acc_Z, micro, uni, hydro, brut = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
files = []
for i in range(3):
    files.append(TdmsFile(paths[i]))

# importation des données
for j in range(len(paths)):
    channel = files[j].object('Untitled', 'Tri100gx_Y_SN2139607')
    acc_Y[j] = channel.data
    channel = files[j].object('Untitled', 'Tri100gy_X_SN2139607')
    acc_X[j] = channel.data
    channel = files[j].object('Untitled', 'Tri100gz_-Z_SN2139607')
    acc_Z[j] = channel.data
    channel = files[j].object('Untitled', 'Micro_SN98656')
    micro[j] = channel.data
    time[j] = channel.time_track()
    channel = files[j].object('Untitled', 'Uni100g_Y_SN4880946')
    uni[j] = channel.data
    channel = files[j].object('Untitled', 'Hydrophone_SN5493417')
    hydro[j] = channel.data
    channel = files[j].object('Untitled', 'AE_Brut_SN5527551')
    brut[j] = channel.data


# concatenate value of same alpha and sigma
Acc_X = np.append(acc_X[0], acc_X[1])
Acc_X = np.append(Acc_X, acc_X[2])

Acc_Y = np.append(acc_Y[0], acc_Y[1])
Acc_Y = np.append(Acc_Y, acc_Y[2])

Acc_Z = np.append(acc_Z[0], acc_Z[1])
Acc_Z = np.append(Acc_Z, acc_Z[2])

Hydro = np.append(hydro[0], hydro[1])
Hydro = np.append(Hydro, hydro[2])

Micro = np.append(micro[0], micro[1])
Micro = np.append(Micro, micro[2])

Brut = np.append(brut[0], brut[1])
Brut = np.append(Brut, brut[2])

Uni = np.append(uni[0], uni[1])
Uni = np.append(Uni, uni[2])

Time = np.linspace(0, 60, num=2400000)

# enregistrement sous forme de dictionnaire 
data = {'Names': Name,
        'Alpha': alpha,
        'Sigma': sigma,
        'Time': Time,
        'Acc_X': Acc_X,
        'Acc_Y': Acc_Y,
        'Acc_Z': Acc_Z,
        'Micro': Micro,
        'Uni': Uni,
        'Hydro': Hydro,
        'Brut': Brut,
        'Cavit': cavitation}

# enregistrement du dictionnaire  en pickle (plus compact que les dataframes)
# f = open('Datas\\Pickle\\df\\' + str(Name) + '_df.pckl', 'wb')
# pickle.dump(df, f)
# f.close()

# fonction pour tester les fichier audios
def sound(track):
    sd.play(track, 44000)
