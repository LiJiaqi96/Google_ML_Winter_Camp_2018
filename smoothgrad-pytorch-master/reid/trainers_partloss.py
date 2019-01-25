from __future__ import print_function, absolute_import
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
from .utils import Bar
from torch.nn import functional as F

class BaseTrainer(object):
    def __init__(self, model, criterion, X, Y, SMLoss_mode=0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(data_loader))
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            #print(inputs)
            inputs, targets = self._parse_data(inputs)
            loss0, loss1 , prec1 = self._forward(inputs, targets)# , loss2, loss3,, loss4, loss5, loss6, loss7
#===================================================================================
            loss = (loss0+loss1)/2#+loss2+loss3+loss4+loss5+loss6+loss7
            losses.update(loss.data, targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()#, loss2, loss3,loss4, loss5, loss6, loss7     , torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda()
            torch.autograd.backward([ loss0, loss1 ],[torch.tensor(1.0).cuda(), torch.tensor(1.0).cuda()])
            optimizer.step()
            #print('Epoch: [{N_epoch}][{N_batch}/{N_size}] | Loss0 {N_loss0:.3f} |Loss1 {N_loss1:.3f}  |Loss2 {N_loss2:.3f} | Loss3 {N_loss3:.3f} |Loss4 {N_loss:.3f} |Loss5 {N_loss:.3f} | Loss {N_loss:.3f} {N_lossa:.3f} |||Prec {N_prec:.2f} {N_preca:.2f}'.format(
             #         N_epoch=epoch, N_batch=i + 1, N_size=len(data_loader),N_loss0=loss0,N_loss1=loss1, N_loss2=loss2,N_loss3=loss3,N_loss4=loss4,N_loss5=loss5,
             #   N_loss=losses.val, N_lossa=losses.avg,N_prec=precisions.val, N_preca=precisions.avg))

            batch_time.update(time.time() - end)
            end = time.time()
            # plot progress
            bar.suffix = 'Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bt:.3f} {N_bta:.3f} | Data {N_dt:.3f} {N_dta:.3f} | Loss {N_loss:.3f} {N_lossa:.3f} | Prec {N_prec:.2f} {N_preca:.2f}'.format(
                      N_epoch=epoch, N_batch=i + 1, N_size=len(data_loader),
                              N_bt=batch_time.val, N_bta=batch_time.avg,
                              N_dt=data_time.val, N_dta=data_time.avg,
                              N_loss=losses.val, N_lossa=losses.avg,
                              N_prec=precisions.val, N_preca=precisions.avg,
							  )
            bar.next()
        bar.finish()



    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError



class Trainer(BaseTrainer):

    def _parse_data(self, inputs):
        imgs, pids,_,_ = inputs
        inputs = [Variable(imgs)]
        #targets = Variable(pids)
        #print(pids)
        targets = Variable(torch.LongTensor(list(map(int, pids))).cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        index = (targets-751).data.nonzero().squeeze_()
        predict = outputs[1][1].data
        prec, = accuracy(predict.data, targets.data)
        prec = prec.item()
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss0 = self.criterion(outputs[1][0],targets)
            loss1 = self.criterion(outputs[1][1],targets)
            # loss2 = self.criterion(outputs[1][2],targets)
            # loss3 = self.criterion(outputs[1][3],targets)
            # loss4 = self.criterion(outputs[1][4],targets)
            # loss5 = self.criterion(outputs[1][5],targets)
            # loss6 = self.criterion(outputs[1][6], targets)
            # loss7 = self.criterion(outputs[1][7], targets)
            #print("-----")
            #print(predict)
            #print(outputs[1][0].size())
            #print(len(outputs[1][0].data),outputs[1][0],outputs[1][0].data,len(outputs[1][1].data),len(targets.data))
            #prec, = accuracy(predict.data, targets.data)

            # print(outputs[1][0],len(outputs[1][0]))
            # print(targets.data)
            # print(prec)
            # print(loss0,loss1.data)

            #prec = prec.item()
                        
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            print(prec)

            prec = prec.item()
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss0, loss1,prec #, loss2, loss3, prec # loss4, loss5, loss6, loss7
