
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

# Paramètres
#############################################################################
filename = 'all_data'
#############################################################################

t0 = datetime.now()
folder_path = 'C:\\Users\\voide\\Documents\\Master\\Machine Learning\\Datas\\Pickle\\dic'
listdir = []
j = 0
buf = []
for i in os.listdir(folder_path):
    listdir.append(i)

len_files = len(listdir)
Data = {}
for j in range(len_files):
    buf = io.open(folder_path + '\\' + listdir[j], 'rb', buffering=(128 * 2048))
    Data[j] = pickle.load(buf)
    buf.close()
    buf = None
    print('Load files : ' + str(listdir[j]))

df = pd.DataFrame(Data)
f = open('Datas\\Pickle\\' + filename + '.pckl', 'wb')
pickle.dump(df, f)
f.close()

#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')
