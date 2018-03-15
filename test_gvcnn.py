# -*- coding: utf-8 -*-
from __future__ import print_function
import datasets
import os

from models import gvcnn
from datasets import modelnet40
from utils import meter
import utils.config

import torch
import torch.utils.data
from torch.autograd import Variable


def test(test_loader, model_gnet, cfg):
    """
    test for one epoch on the testing set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    mAP = meter.mAPMeter()

    # training mode
    model_gnet.eval()

    for i, (shapes, labels) in enumerate(test_loader):
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
        preds = model_gnet(shapes)
        labels_oh = torch.zeros(labels.data.size(0), cfg.class_num)\
            .scatter_(1, labels.cpu().data.unsqueeze(1), 1)
        mAP.add(preds.data, labels_oh)

        prec.add(preds.data, labels.data)

        if i % cfg.print_freq == 0:
            print('[{0}/{1}]\t'
                  'Batch Time {batch_time:.3f}\t'
                  'Epoch Time {data_time:.3f}\t'
                  'Prec@1 {top1:.3f}\t'.format(
                 i, len(test_loader), batch_time=batch_time.value(),
                data_time=data_time.value(), top1=prec.value(1)))

    print('mean class accuracy : {0} '.format(prec.value(1)))
    print('mAP: %f' % mAP.value())



def main():
    cfg = utils.config.config()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    test_dataset = datasets.modelnet40.Modelnet40_dataset(cfg, status='test')

    print('number of test samples is: ', len(test_dataset))

    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size,
                                              shuffle=True, num_workers=cfg.workers)
    # create model
    model = gvcnn.GVCNN(pretrained=True, aux_logits=cfg.have_aux,
                                   transform_input=False, num_classes=40,
                                   n_views=cfg.data_views, with_group=cfg.with_group)
    print('loading model from {0}'.format(cfg.ckpt_model))
    checkpoint = torch.load(cfg.ckpt_model)
    model.load_state_dict(checkpoint['model_param_best'])

    print('GVCNN: ')
    print(model)

    if cfg.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()

    test(val_loader, model, cfg)

if __name__ == '__main__':
    main()

