from whale.forms import ProfileForm 
from whale.models import Profile 
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
import torch
from reid import models
from torch.autograd import Variable
import numpy as np
import cv2
import pandas as pd
import os

cuda = False
voc = {}
def readLabel():
    lable = pd.read_csv('/root/HumpbackWhale/label.csv')
    lable.drop_duplicates('newId',inplace=True)
    for k, v in zip(lable['newId'],lable['Id']):
         voc[k] = v

readLabel()

def find_new_img():
    dir = '/root/HumpbackWhale/whale/whale/static/uploads/test_pictures'
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + "/" + fn) if not os.path.isdir(dir + "/" + fn) else 0)
    return os.path.join(dir, file_lists[-1])

def preprocess_image(img, cuda=False):
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    if cuda:
        preprocessed_img = Variable(preprocessed_img.cuda(), requires_grad=True)
    else:
        preprocessed_img = Variable(preprocessed_img, requires_grad=True)

    return preprocessed_img

def predict():
    img = preprocess_image(cv2.imread(find_new_img(), 1))
    model = models.create('resnet50', num_features=256,
                          dropout=0.25, num_classes=5005,cut_at_pooling=False, FCN=True)
    tar = torch.load('/root/HumpbackWhale/checkpoint.pth.tar',map_location='cpu')
    state_dict = tar['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
       if 'module.' in k:
           name = k[7:] # remove `module.`
       else:
           name = k
       new_state_dict[name] = v
    model.load_state_dict(new_state_dict)#{'state_dict':new_state_dict})

    #model.load_state_dict(torch.load("/root/HumpbackWhale/identification/checkpoint.pth.tar",map_location='cpu'))
    if cuda:
        model = model.cuda()
    d  = model(img)
    #return int(torch.max(model(img)[1][0][0], dim=0)[1].numpy())
    return voc[int(torch.max(model(img)[1][0][0], dim=0)[1].numpy())]
    #return torch.max(model(img)[1][0], dim=0)[1]

@csrf_exempt 
def index(request):
    context = {}
    form = ProfileForm
    context['form'] = form 
    return render(request, 'index.html', context)

@csrf_exempt 
def save_profile(request):
    if request.method == "POST":
        form = ProfileForm(request.POST, request.FILES)
        if form.is_valid():
            profile = Profile()
            profile.picture = form.cleaned_data["picture"]
            profile.save()
            return HttpResponse(predict())
        else:
            return HttpResponse('Error')
    return HttpResponse('Error! POST')







