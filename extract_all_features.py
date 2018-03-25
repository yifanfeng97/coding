# -*- coding: utf-8 -*-
from __future__ import print_function
import datasets
import os
import h5py
import numpy as np
import sys
sys.path.append('models')
sys.path.append('datasets')

from models import gvcnn
from datasets import modelnet40
from utils import meter
import utils.config

import torch
import torch.utils.data
from torch.autograd import Variable


def get_feature(cfg, data_loader, model):
    """
    test for one epoch on the testing set
    """



    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    mAP = meter.mAPMeter()

    feature_all = None
    label_all = None

    # training mode
    model.eval()

    for i, (shapes, labels) in enumerate(data_loader):
        batch_time.reset()
        # bz x 12 x 3 x 224 x 224
        labels = labels.long().view(-1)
        shapes = Variable(shapes)
        labels = Variable(labels)

        # shift data to GPU
        if cfg.cuda:
            shapes = shapes.cuda()
            labels = labels.cuda()

        # forward, backward optimize
        preds, features = model(shapes)
        labels_oh = torch.zeros(labels.data.size(0), cfg.class_num)\
            .scatter_(1, labels.cpu().data.unsqueeze(1), 1)
        mAP.add(preds.data, labels_oh)
        prec.add(preds.data, labels.data)

        features = features.data.cpu().numpy()
        labels = labels.data.cpu().numpy().reshape((-1, 1))

        if feature_all is None:
            feature_all = features
        else:
            feature_all = np.vstack((feature_all, features))

        if label_all is None:
            label_all = labels
        else:
            label_all = np.vstack((label_all, labels))

        if i % cfg.print_freq == 0:
            print('[{0}/{1}]\t'
                  'Batch Time {batch_time:.3f}\t'
                  'Epoch Time {data_time:.3f}\t'
                  'Prec@1 {top1:.3f}\t'.format(
                 i, len(data_loader), batch_time=batch_time.value(),
                data_time=data_time.value(), top1=prec.value(1)))

    print('mean class accuracy : {0} '.format(prec.value(1)))
    print('mAP: %f' % mAP.value())

    return feature_all, label_all




def main():
    cfg = utils.config.config()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

    f = h5py.File(cfg.feature_file, 'w')

    train_dataset = datasets.modelnet40.Modelnet40_dataset(cfg, status='train')
    val_dataset = datasets.modelnet40.Modelnet40_dataset(cfg, status='val')
    test_dataset = datasets.modelnet40.Modelnet40_dataset(cfg, status='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size,
                                              shuffle=False, num_workers=cfg.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size,
                                              shuffle=False, num_workers=cfg.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size,
                                              shuffle=False, num_workers=cfg.workers)

    train_len = len(train_dataset)+len(val_dataset)
    test_len = len(test_dataset)

    f.create_group('train')
    f.create_group('test')

    f['train'].create_dataset('features', shape=(train_len, 2048))
    f['train'].create_dataset('labels', shape=(train_len, 1))

    f['test'].create_dataset('features', shape=(test_len, 2048))
    f['test'].create_dataset('labels', shape=(test_len, 1))

    print('number of test samples is: ', len(test_dataset))

    # create model
    model = gvcnn.GVCNN(pretrained=True, aux_logits=cfg.have_aux, get_feature=True,
                                   transform_input=False, num_classes=40,
                                   n_views=cfg.data_views, with_group=cfg.with_group)
    print('loading model from {0}'.format(cfg.ckpt_model))
    checkpoint = torch.load(cfg.ckpt_model)
    model.load_state_dict(checkpoint['model_param_best'])

    # print('GVCNN: ')
    # print(model)

    if cfg.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()

    train_f, train_l = get_feature(cfg, train_loader, model)
    val_f, val_l = get_feature(cfg, val_loader, model)
    test_f, test_l = get_feature(cfg, test_loader, model)

    f['train']['features'][:] = np.vstack((train_f, val_f))
    f['train']['labels'][:] = np.vstack((train_l, val_l))

    f['test']['features'][:] = test_f
    f['test']['labels'][:] = test_l

    f.close()


if __name__ == '__main__':
    main()

