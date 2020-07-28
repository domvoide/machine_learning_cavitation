# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:39:52 2020

@author: voide
"""

import pandas as pd
import os
from datetime import datetime
import sounddevice as sd
import pickle

t0 = datetime.now()
 
path = 'Datas\\Centrale\\Data_PA\\'
listfiles = os.listdir(path)
dfAll = pd.DataFrame()
for files in listfiles[1::]:
    filename = files
    xls = pd.ExcelFile(path + filename)
    df = xls.parse('Sheet1', skiprows=9, usecols=[1], na_values=['NAN'])
    df.rename(columns={'Micro_Turbine_SN98656':str(filename[-8:-5])}, inplace=True)
    print(df.head(4))
    dfAll = pd.concat([dfAll, df], axis=1)

print(dfAll.head(4))

# fonction pour tester les fichier audios
def sound(track):
    sd.play(track, 10000)

f = open('Datas\\Pickle\\Centrale.pckl', 'wb')
pickle.dump(dfAll, f)
f.close()


#affichage du temps écoulé
t1 = datetime.now()
print('Temps écoulé : ' + str(t1-t0) + ' [h:m:s]')
    
