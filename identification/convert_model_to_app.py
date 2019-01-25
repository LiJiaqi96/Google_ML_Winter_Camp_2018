from __future__ import print_function, absolute_import
from torch.autograd import Variable
import torch.onnx
import torchvision

import argparse

import os
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
import os.path as osp
from reid import models
from reid.utils.serialization import load_checkpoint, save_checkpoint


# Create model
model = torch.load("model.pth").cuda()

#model = nn.DataParallel(model).cuda()
dummy_input = Variable(torch.randn(32, 3, 256, 256)).cuda()


torch.onnx.export(model, dummy_input, "alexnet.proto", verbose=True)
