import torch
import pandas as pd
import collections


df = pd.read_csv("../dataset/label.csv")

df = df.sample(frac=1)
cut_idx = int(round(0.2 * df.shape[0]))
df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
print(len(df),df_test,df_train)

