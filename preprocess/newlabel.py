import numpy as np
import pandas as pd

data = pd.read_csv('../dataset/label.csv')

import os
folder = "../newdataset/train/"

res = []
for filename in os.listdir(folder):
    d = data[data['Image'] == filename.split('_')[0]+'.jpg']
    newid = d['newId'].values[0]
    id = d['Id'].values[0]
    res.append([filename, id, newid])

from pandas.core.frame import DataFrame
df = pd.DataFrame(res, columns=['Image', 'Id', 'newId']) 
df.to_csv('../dataset/newlabel.csv')

#label = data['Id'].drop_duplicates()

#ata['newId'] = data['Id']
#for l, i in zip(label, range(len(label))):
#   data.loc[data['newId']==l,'newId'] = i

#data.to_csv('../dataset/label.csv',index= False)

     
