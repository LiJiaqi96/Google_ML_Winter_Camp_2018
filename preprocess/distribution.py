# Data distribution
import pandas as pd
import numpy as np

info = pd.read_csv('../label.csv')
types = info['newId']
names = info['Id']
type_count_dict = {}
type_name_dict = {}
for i in range(len(types)):
    if(types[i] in type_count_dict.keys()):
        type_count_dict[types[i]] += 1
    else:
        type_count_dict[types[i]] = 0
        type_name_dict[types[i]] = names[i]
        
classes = list(type_name_dict.values())
counts = list(type_count_dict.values())

print('print')
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')
plt.figure(figsize=(10,10))
plt.title('Distribution of Different Types of Whales')
plt.pie(counts, colors=plt.cm.Blues(1.*np.arange(len(classes))[::-1]/len(classes)),
        shadow=False,startangle=90)
# autopct='%1.1f%%',

#plt.show()#.save('a.jpg')
# 47.5%

plt.savefig('distribution.jpg')
