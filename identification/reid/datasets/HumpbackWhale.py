from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re
import pandas as pd
from PIL import Image
import os

class LabelOneHotEncoder():
    def __init__(self):
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()
    def fit_transform(self, x):
        features = self.le.fit_transform( x)
        return self.ohe.fit_transform( features.reshape(-1,1))
    def transform( self, x):
        return self.ohe.transform( self.la.transform( x.reshape(-1,1)))
    def inverse_tranform( self, x):
        return self.le.inverse_transform( self.ohe.inverse_tranform( x))
    def inverse_labels( self, x):
        return self.le.inverse_transform( x)

class HW_Dataset(object):
    def __init__(self, filepath, csv_path, transform=None):
        self.file_path = filepath
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_list = [x for x in os.listdir(self.file_path)]
        self.train, self.test = [], [], []
    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, idx):


        img_path = os.path.join(self.file_path, self.df.Image[idx])
        label = self.df.Id[idx]
        new_label = self.df.newId[idx]

        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs = Image.open(img_path)

        imgs = imgs.convert('RGB')
        imgs = self.transform(imgs)

        return imgs, new_label



