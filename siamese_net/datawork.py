import numpy as np
import pandas as pd
import random
import sys
from skimage import io, transform, color
from matplotlib import pyplot as plt

def img_re_co(img,height,width):
    img_re = transform.resize(img, (height, width))
    if (len(img_re.shape) == 2):
        img_co = color.gray2rgb(img_re)
    else:
        img_co = img_re
    return img_co

# Loading the original info
# train_info = pd.read_csv('/Users/ljq/Desktop/trainlabel.csv')
# val_info = pd.read_csv('/Users/ljq/Desktop/validationlabel.csv')
# path = "/Users/ljq/Desktop/whale_data/train/"

# train_info = pd.read_csv('/data8t/ljq/whale_data/whale_data/trainlabel.csv')
# val_info = pd.read_csv('/data8t/ljq/whale_data/whale_data/validationlabel.csv')
# path = "/data8t/ljq/whale_data/whale_data/train/"

train_info = pd.read_csv('/home/jqli1403/HumpbackWhale/dataset/trainlabel.csv')
val_info = pd.read_csv('/home/jqli1403/HumpbackWhale/dataset/validationlabel.csv')
path = "/home/jqli1403/HumpbackWhale/dataset/train/"

train_types = train_info['newId']
train_names = train_info['Image']
val_types = val_info['newId']
val_names = val_info['Image']

# Uniform size of images
height = 256
width = 256

# Construct the type-name dictionary for training data
train_name_dict = {}
for i in range(len(train_types)):
    if(train_types[i] in train_name_dict.keys()):
        train_name_dict[train_types[i]].append(train_names[i])
    else:
        train_name_dict[train_types[i]] = [train_names[i]]
# Raw labels
train_base_index = train_name_dict.keys()
train_base_num = len(train_name_dict.keys())
print(train_base_num)

# Construct the training set
train_data = [np.zeros((train_base_num*4,height,width,3)) for j in range(2)]
train_label = np.zeros((train_base_num*4,1))
# Sample-pair with same type (label 1)
for idx in train_base_index:
    print(idx)
    if(len(train_name_dict[idx]) == 1):
        img = io.imread(path+train_name_dict[idx][0])
        img_co = img_re_co(img,height,width)
        train_data[0][idx, :, :, :] = img_co
        train_data[1][idx, :, :, :] = img_co
        train_label[idx] = 1
    else:
        # print(train_name_dict[idx])
        no1 = random.randint(0,len(train_name_dict[idx])-1)
        no2 = random.randint(0,len(train_name_dict[idx])-1)
        img1 = io.imread(path + train_name_dict[idx][no1])
        img2 = io.imread(path + train_name_dict[idx][no2])
        img_co1 = img_re_co(img1, height, width)
        img_co2 = img_re_co(img2, height, width)
        # plt.imshow(img_co1)
        # plt.show()
        train_data[0][idx, :, :, :] = img_co1
        train_data[1][idx, :, :, :] = img_co2
        train_label[idx] = 1
# Sample-pair with different type (label 0)
# The number of pairs here is 5*number of pairs with same type
count = train_base_num
for k in range(train_base_num):
    cnt = 1
    while(cnt <= 3):
        no1 = random.randint(0,len(train_name_dict[k])-1)
        index = random.randint(0,train_base_num-1)
        if(index == count):
            continue
        else:
            no2 = random.randint(0,len(train_name_dict[index])-1)
            img1 = io.imread(path + train_name_dict[k][no1])
            img2 = io.imread(path + train_name_dict[index][no2])
            img_co1 = img_re_co(img1, height, width)
            img_co2 = img_re_co(img2, height, width)
            train_data[0][count, :, :, :] = img_co1
            train_data[1][count, :, :, :] = img_co2
            train_label[count] = 0
            cnt += 1
            count += 1
            print(count)
# Shuffle training data
indexs = np.arange(len(train_data[0]))
np.random.shuffle(indexs)
train_data[0] = train_data[0][indexs]
train_data[1] = train_data[1][indexs]
train_label = train_label[indexs]

# print(train_data.shape)
print(sys.getsizeof(train_data))

np.save("training_data.npy",train_data)
np.save("training_label.npy",train_label)

# Construct the validation set
val_name_dict = {}
for i in range(len(val_types)):
    if(val_types[i] in val_name_dict.keys()):
        val_name_dict[val_types[i]].append(val_names[i])
    else:
        val_name_dict[val_types[i]] = [val_names[i]]
# Raw labels
val_base_index = val_name_dict.keys()
val_base_num = len(val_name_dict.keys())
print(val_base_num)

val_data = [np.zeros((val_base_num*3,height,width,3)) for j in range(2)]
val_label = np.zeros((val_base_num*3,1))
# Sample-pair with same type (label 1)
for idx in val_base_index:
    print(idx)
    if(len(val_name_dict[idx]) == 1):
        img = io.imread(path+val_name_dict[idx][0])
        img_co = img_re_co(img,height,width)
        val_data[0][idx, :, :, :] = img_co
        val_data[1][idx, :, :, :] = img_co
        val_label[idx] = 1
    else:
        no1 = random.randint(0,len(val_name_dict[idx])-1)
        no2 = random.randint(0,len(val_name_dict[idx])-1)
        img1 = io.imread(path + val_name_dict[idx][no1])
        img2 = io.imread(path + val_name_dict[idx][no2])
        img_co1 = img_re_co(img1, height, width)
        img_co2 = img_re_co(img2, height, width)
        val_data[0][idx, :, :, :] = img_co1
        val_data[1][idx, :, :, :] = img_co2
        val_label[idx] = 1
# Sample-pair with different type (label 0)
# The number of pairs here is 5*number of pairs with same type
count_ = val_base_num
for k in range(val_base_num):
    cnt = 1
    while(cnt <= 2):
        no1 = random.randint(0,len(val_name_dict[k])-1)
        index = random.randint(0,val_base_num-1)
        if(index == count_):
            continue
        else:
            no2 = random.randint(0,len(val_name_dict[index])-1)
            img1 = io.imread(path + val_name_dict[k][no1])
            img2 = io.imread(path + val_name_dict[index][no2])
            img_co1 = img_re_co(img1, height, width)
            img_co2 = img_re_co(img2, height, width)
            val_data[0][count_, :, :, :] = img_co1
            val_data[1][count_, :, :, :] = img_co2
            val_label[count_] = 0
            cnt += 1
            count_ += 1
            print(count_)
# Shuffle validation data
indexs = np.arange(len(val_data[0]))
np.random.shuffle(indexs)
val_data[0] = val_data[0][indexs]
val_data[1] = val_data[1][indexs]
val_label = val_label[indexs]

# print(val_data.shape)
print(sys.getsizeof(val_data))

np.save("validation_data.npy",val_data)
np.save("validation_label.npy",val_label)

print(len(train_types))
print(len(train_names))
print(len(val_types))
print(len(val_names))

# Construct a small t
