import numpy as np
import os
import copy
import keras.backend as K
from skimage import transform, color, io
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers import Conv2D, Flatten, Dense, Subtract, Lambda, MaxPooling2D, UpSampling2D
from scipy import interp
from sklearn import metrics
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan)

# Function of obtaining the feature map
def get_encoded(model,input_img,layer_num):
    # input: image, which is the same as the input of model : [model.layers[0].input]
    feature_map = K.function(input_img, [model.layers[layer_num].output])
    return feature_map

def img_re_co(img,height,width):
    img_re = transform.resize(img, (height, width))
    if (len(img_re.shape) == 2):
        img_co = color.gray2rgb(img_re)
    else:
        img_co = img_re
    return img_co

# Training parameters
model_name = 'AE_Cluster'
height = 256
width = 256
input_shape = (256,256,3)
batch_size = 32
epochs = 100

# Data
train_path = "/data8t/ljq/whale_data/whale_data/train/"
test_path = "/data8t/ljq/whale_data/whale_data/test/"
train_list = os.listdir(train_path)
test_list = os.listdir(test_path)

train_data = np.zeros(len(train_list),height,width,3)
for i in range(len(train_list)):
    img = io.imread(train_path + train_list[i])
    img_co = img_re_co(img, height, width)
    train_data[i, :, :, :] = img_co
train_label = copy.deepcopy(train_data)

test_data = np.zeros(len(test_list),height,width,3)
for i in range(len(test_list)):
    img = io.imread(test_path + test_list[i])
    img_co = img_re_co(img, height, width)
    test_data[i, :, :, :] = img_co

# Model
model = Sequential()

# Encode
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(8, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(1, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

# Decode
model.add(Conv2D(1, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding='same'))
model.add(UpSampling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(8, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding='same'))
model.add(UpSampling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding='same'))
model.add(UpSampling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding='same'))
model.add(UpSampling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(3, kernel_size=(3,3), activation='sigmoid', input_shape=input_shape, padding='same'))

model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.summary()

model.fit(train_data, train_label, batch_size=batch_size, epochs= epochs, verbose=1)

model_json = model.to_json()

# Save model
if not os.path.exists('./models'):
    os.makedirs('./models')
with open('models/' + model_name + '.json', "w") as json_file:
    json_file.write(model_json)
model.save_weights('models/' + model_name + '.h5')

features = np.zeros(len(test_list),height,width)
# Construct the featuremap data
for j in range(len(test_list)):
    features[j,:,:] = get_encoded(model,test_data[j,:,:,:],5)
np.save("feature_maps.npy",features)


