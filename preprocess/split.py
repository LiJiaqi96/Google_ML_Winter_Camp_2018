from sklearn.model_selection import train_test_split

import pandas as pd

data = pd.read_csv('../dataset/train.csv')

d_train, d_test = train_test_split(data,test_size=0.2)

d_train.to_csv('../dataset/newtrain.csv', index=False)
d_test.to_csv('../dataset/validation.csv', index=False)

