import numpy as np
import pandas as pd

data = pd.read_csv('../identification/label2.csv')

label = data['Id'].drop_duplicates()

data['newId'] = data['Id']
for l, i in zip(label, range(len(label))):
   data.loc[data['newId']==l,'newId'] = i

data.to_csv('../dataset/label2.csv',index= True)

     
