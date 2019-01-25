import cv2
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

import random

def aug(images):
	noise = random.random()/8
	blur = random.random()
	contrast = random.random()/2+1
	rotate = random.random()*10
	seq = iaa.SomeOf((1, None), [
	        iaa.AdditiveGaussianNoise(scale=noise*255),
	        iaa.GaussianBlur(sigma=blur),
	        iaa.Grayscale(alpha=1.0),
	        iaa.GammaContrast(gamma=contrast),
	        iaa.Fliplr(1.0),
	        iaa.Affine(rotate=rotate),
		iaa.Affine(rotate=360-rotate),
	], random_order=True)
	return seq.augment_images(images)

def flip(images):
	seq = iaa.Sequential([iaa.Fliplr(1.0)])
	return seq.augment_images(images)

from glob import glob
import scipy.misc
import os
folder = '../dataset/train/'
savefolder = '../newdataset/train/'
for filename in os.listdir(folder):
	img = cv2.imread(os.path.join(folder,filename))
	if img is not None:
		augimg = flip([img])
		fn = filename.split('.')
		fn = fn[0]+'_0.'+fn[1]
		scipy.misc.imsave(os.path.join(savefolder,fn), augimg[0])
		for i in range(1,int(random.random()*10+1)):
			augimg = aug([img])
			fn = filename.split('.')
			fn = fn[0]+'_'+str(i)+'.'+fn[1]
			scipy.misc.imsave(os.path.join(savefolder,fn), augimg[0])
