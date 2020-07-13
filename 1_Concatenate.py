
"""
Created on Wed Feb 20 13:42:05 2020

@author: dominiqu.voide

Mise en commun des fichiers pickle créés pour chaque point de fonctionnement 
afin d'avoir un seul fichier contenant toutes les informations

"""

import os
import io
import pickle
import pandas as pd
from datetime import datetime

# Paramètres à modifier
#############################################################################
filename = 'all_data'
#############################################################################

t0 = datetime.now()

# dossier où tous les précédents fichiers ont été importé avec le fichier 0_Import
folder_path = 'C:\\Users\\voide\\Documents\\GitHub\\machine_Learning_cavitation\\Datas\\Pickle\\dic'
listdir = []
j = 0
buf = []

# création de la liste de tous les fichiers contenu dans le dossier
for i in os.listdir(folder_path):
    listdir.append(i)

# extraction des valeurs sous forme de dictionnaire pour chaque fichier
len_files = len(listdir)
Data = {}
for j in range(len_files):
    buf = io.open(folder_path + '\\' + listdir[j], 'rb', buffering=(128 * 2048))
    Data[j] = pickle.load(buf)
    buf.close()
    buf = None
    print('Load files : ' + str(listdir[j]))

# mise en dataframe et enregistrement en pickle
df = pd.DataFrame(Data)
f = open('Datas\\Pickle\\' + filename + '.pckl', 'wb')
pickle.dump(df, f)
f.close()

#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')
