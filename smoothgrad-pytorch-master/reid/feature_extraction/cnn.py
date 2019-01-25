from __future__ import absolute_import
from collections import OrderedDict

import scipy.io as sio
from torch.autograd import Variable
import torch
from ..utils import to_torch


def extract_cnn_feature(model, inputs, modules=None, return_mask = False):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs, volatile=True)
    if modules is None:
        tmp = model(inputs)
       # sio.savemat('PCB_query_result.mat',{'query_f'})
        outputs = tmp[0]
       # outputs = torch.stack((tmp[1][1],tmp[1][2],tmp[1][3],tmp[1][4]),dim =2)
        outputs = outputs.data.cpu()
        if return_mask:
            mask = tmp[4]
            mask = mask.data.cpu()
            return outputs, mask
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())
