import numpy as np
import pickle
import io
from scipy.io.wavfile import write

output_path = 'C:\\Users\\voide\\Documents\\Master\\Machine Learning\
\Documents\\Analyse_features\\Audio\\'
# importation du pickle contenant les 18 fichiers de données
folder_path = 'Datas\\Pickle\\all_data.pckl'
buf = io.open(folder_path, 'rb', buffering=(128 * 2048))
df = pickle.load(buf)
buf.close()
del buf
df = df.transpose()
namenumber = 4
file = df.Micro[namenumber]

f = 40000

name = 'Audio_all_' + str(namenumber)
m = np.max(np.abs(file))
sigf32 = (file/m).astype(np.float32)
write(output_path + name + '.wav', f, sigf32)