from __future__ import absolute_import
import os.path as osp
import pandas as pd
import os
from PIL import Image

class HW_Dataset(object):
    def __init__(self, filepath, csv, transform=None):
        self.file_path = filepath
        self.df = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return (len(self.df))

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, idx):
        #print("idx:",idx,self.__len__())
        #print("image:",self.df.Image[idx])
        #print("id:",self.df.Id[idx])
        #print("new_id:",self.df.newId[idx])
        #print("idx:",self.df.index[idx])
        img_name = self.df.Image[idx]
        img_path = os.path.join(self.file_path, self.df.Image[idx])
        label = self.df.Id[idx]
        new_label = self.df.newId[idx]
        index = self.df.index[idx]
        # img = cv2.imread(img_path)
        #print("t(++++")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs = Image.open(img_path)

        imgs = imgs.convert('RGB')
        imgs = self.transform(imgs)
        #print("____---")
        return imgs, new_label, label,index

class HW_Test_Dataset(object):
    def __init__(self, filepath, csv, transform=None):
        self.file_path = filepath
        self.df = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return (len(self.df))

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, idx):
        img_name = self.df.Image[idx]
        img_path = os.path.join(self.file_path, img_name)
        label = self.df.Id[idx]
        new_label = self.df.Id[idx]

        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #imgs = self.image_list[idx]#Image.open(img_path)
        imgs =  Image.open(img_path)
        imgs = imgs.convert('RGB')
        imgs = self.transform(imgs)

        return imgs, new_label, label,label

class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid
