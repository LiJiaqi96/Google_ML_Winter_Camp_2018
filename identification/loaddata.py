from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import pandas as pd

from reid import datasets
from reid import models
from reid.trainers_partloss import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import HW_Dataset,HW_Test_Dataset
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint



#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def get_data(dataset_dir, height, width, batch_size, workers):

    train_filepath = osp.join(dataset_dir,'train/')
    train_csv_path = osp.join(dataset_dir,'label.csv')
    test_filepath = osp.join(dataset_dir,'test/')
    test_csv_path = osp.join(dataset_dir,'test.csv')

    #df_test = pd.read_csv("../dataset/label1.csv")
    #df_train = pd.read_csv("../dataset/label2.csv")
    #df = df.sample(frac=1)
    #cut_idx = int(round(0.2 * df.shape[0]))
    #df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
    #df_test.to_csv("label1.csv",index=0)
    #df_train.to_csv("label2.csv",index =0)
    #df_test =pd.read_csv("label1.csv")
    #df_train = pd.read_csv("label2.csv")
    #print(df_test,df_train)=
    # df = pd.read_csv("../newdataset/newlabel.csv")
    # df = df.sample(frac=1)
    # cut_idx = int(round(0.2 * df.shape[0]))
    # df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
    # df_test.to_csv("label1.csv",index=0)
    # df_train.to_csv("label2.csv",index =0)
    # df_test =pd.read_csv("label1.csv")
    # df_train = pd.read_csv("label2.csv")
    #print(df_test,df_train)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.RectScale(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])
    train_loader = DataLoader(
        HW_Dataset(train_filepath,train_csv_path, transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=False)
    #print(test_dataset)

    test_loader = DataLoader(
        HW_Test_Dataset(test_filepath, test_csv_path, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True, drop_last=False)
    #
    # gallery_loader = DataLoader(
    #     HW_Dataset(test_filepath, csv_path, transform=test_transformer),
    #     batch_size=batch_size, num_workers=workers,
    #     shuffle=False, pin_memory=True)
    #num_classes = df_test['Id'].drop_duplicates()

    return train_loader, test_loader#, len(num_classes)#, gallery_loader


def  main(args):
    # df = pd.read_csv('../dataset/label.csv')
    # num_classes = max(list(map(int,df.newId[1])))
    # print(num_classes)
    num_classes = 5005
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #num_classes
    #device_ids = [0, 1, 2, 3]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (256, 256)

    #num_classes = 5005#get by df['Id'].nunique()
    train_loader, test_loader = \
        get_data(args.data_dir, args.height,
                 args.width, args.batch_size, args.workers,
                 )   #, test_loader

    print(num_classes)
    #num_classes = num_classes
    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes,cut_at_pooling=False, FCN=True)
    #torch.save(model, 'model.pth')
    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model_dict = model.state_dict()
        checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        model_dict.update(checkpoint_load)
        model.load_state_dict(model_dict)
#        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))

    #model = nn.DataParallel(model)
    model = nn.DataParallel(model).cuda()
    #model = model.cuda()


    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test:")
        evaluator.evaluate( train_loader, test_loader,  dataset.query, dataset.gallery)
        return

    # Criterion
    #criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer
    if hasattr(model.module, 'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    torch.save(model,"new_model.pth")
    model = torch.load("new_model.pth")
    # Trainer
    trainer = Trainer(model, criterion, 0, 0, SMLoss_mode=0)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 60 if args.arch == 'inception' else args.step_size
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)#if lr_mult do not find,return defualt value 1

    #Start training
    # for epoch in range(start_epoch, args.epochs):
    #     adjust_lr(epoch)
    #     trainer.train(epoch, train_loader, optimizer)
    #     is_best = True
    #     save_checkpoint({
    #         'state_dict': model.module.state_dict(),
    #         'epoch': epoch + 1,
    #         'best_top1': best_top1,
    #     }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir,'checkpoint.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    #torch.save(model, 'model.pth')
    # model1 = torch.load("model.pth")
    query = pd.read_csv('../dataset/test.csv')
    gallery = pd.read_csv('../dataset/label.csv')
    #print(len(query),len(gallery),len(os.listdir("../dataset/train")),len(os.listdir("../dataset/test")))
    evaluator.evaluate(test_loader, train_loader, query, gallery)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                        "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
            help="train and val sets together for training, "
                             "val set alone for validation")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--step-size',type=int, default=40)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../dataset'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
