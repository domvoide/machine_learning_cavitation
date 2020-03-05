import os
import io
import pickle

import pandas as pd
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
f = open('Datas\\Pickle\\all_data.pckl', 'wb')
pickle.dump(df, f)
f.close()

print(df)
