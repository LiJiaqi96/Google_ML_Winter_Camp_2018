import os
folder = "../dataset/test/"
data = ['Image,Id']
data.append()
i = 0
for filename in os.listdir(folder):
    data.append(filename + ',' + str(i))
    i = i + 1

import numpy as np
f = open('../dataset/test.csv','wb')
np.savetxt(f, np.asarray(data), fmt="%s")
