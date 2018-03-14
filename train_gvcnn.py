# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import datasets
import random
import time
import os
import numpy as np

from models import gvcnn
from datasets import modelnet40
from utils import meter
import utils.config
import utils.meter.averagevaluemeter
import utils.meter.confusionmeter
import train_helper

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


from torch.autograd import Variable



def train(train_loader, model_gnet, criterion, optimizer, epoch, opt):
    """
    train for one epoch on the training set
    """
    batch_time = meter.timemeter.TimeMeter()
    data_time = meter.timemeter.TimeMeter()
    losses = meter.averagevaluemeter.AverageValueMeter()
    prec = meter.classerrormeter.ClassErrorMeter(topk=[1], accuracy=True)
    #############################################
    ## confusion table
    #############################################
    confusion = utils.meter.confusionmeter.ConfusionMeter(40)

    # training mode
    model_gnet.train()

    end = time.time()
    for i, (inputs_12v, labels) in enumerate(train_loader):
        # bz x 12 x 3 x 224 x 224
        # re-view it to be : (bz * 12) x 3 x 224 x 224
        # note that inputs_12v.size(2) = 1, so we expand it to be 3
        inputs_12v = inputs_12v.view(inputs_12v.size(0) * inputs_12v.size(1), inputs_12v.size(2),
                                     inputs_12v.size(3), inputs_12v.size(4))
        labels = labels.long().view(-1)
        if isinstance(inputs_12v, torch.ByteTensor):
            inputs_12v = inputs_12v.float()

        inputs_12v = Variable(inputs_12v)
        labels = Variable(labels)

        # print(points.size())
        # print(labels.size())
        # shift data to GPU
        if opt.cuda:
            inputs_12v = inputs_12v.cuda()
            #            labels = labels.long().cuda() # must be long cuda tensor
            labels = labels.cuda()  # must be long cuda tensor

        # forward, backward optimize
        # (bz*12) X C x H x W

        preds = model_gnet(inputs_12v)  # bz x C x H x W
        #        print('labels:\n', labels.data)
        #        print('preds:\n', preds.data)
        # in pytorch, unlike torch, the label is 0-indexed (start from 0)
        #        labels = labels.sub_(1)

        # debug_here()
        # if labels.data.max() >= 40:
        #    debug_here()
        #   print('error')

        # if labels.data.min() < 0:
        #    debug_here()
        #    print('error')
        ###########################################
        ## add center loss
        ###########################################
        # 40 classes
        #        alpha = 0.1
        #        # preds as features
        #        center_loss, model_after_pool._buffers['centers'] = utils.get_center_loss(model_after_pool._buffers['centers'],
        #            preds, labels, alpha, 40)
        #
        #        # contrastive center loss
        #        print('size')
        #        print(preds.size(), labels.size())
        if opt.have_aux:
            preds, aux = preds
            loss_main = criterion(preds, labels)
            loss_aux = criterion(aux, labels)
            softmax_loss = loss_main + 0.3 * loss_aux
        else:
            softmax_loss = criterion(preds, labels)
        # center_loss_weight = 0 # set it to 0.1, can achive 91.625, set it to 0,
        #        loss = center_loss_weight * center_loss + softmax_loss
        loss = softmax_loss

        ###########################################
        ## measure accuracy
        ###########################################
        prec1 = utils.accuracy(preds.data, labels.data, topk=(1,))[0]
        losses.add(loss.data[0], preds.size(0))  # batchsize
        prec.add(prec1[0], preds.size(0))

        ###############################################
        ## confusion table
        ###############################################
        #        print('preds:')
        #        print(preds.data)
        #        print('labels:')
        #        print(labels.data)
        confusion.batchAdd(preds.data, labels.data)

        ###########################################
        ## backward
        ###########################################
        optimizer.zero_grad()
        loss.backward()
        #        utils.clip_gradient(optimizer, opt.gradient_clip)
        optimizer.step()

        # debug_here()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))
            # ###########################################
            # ## Log
            # ###########################################
            # # loss accuracy
            # step = epoch * len(train_loader) + i
            # loss_name= None
            # if opt.have_aux:
            #     loss_name = 'mixed_loss'
            # else:
            #     loss_name = 'loss'
            # info = { loss_name : loss.data[0],
            #         'accuracy': top1.avg}
            #
            # for tag, value in info.items():
            #     logger.scalar_summary(log_pre_name+'train/' + tag, value, step)
            # # parameters gradients
            # for tag, value in model_gnet.named_parameters():
            #     if not hasattr(value.grad, 'data'): continue
            #     tag = tag.replace('.', '/')
            #     logger.histo_summary(log_pre_name+'train/' + tag, to_np(value), step)
            #     logger.histo_summary(log_pre_name+'train/' + tag + '/grad', to_np(value.grad), step)
            # # images
    #####################################
    ## confusion table
    #####################################
    # debug_here()
    confusion.updateValids()
    print('mean class accuracy at epoch {0}: {1} '.format(epoch, confusion.mean_class_acc))


def validate(test_loader, model_gnet, criterion, optimizer, epoch, opt):
    """
    test for one epoch on the testing set
    """
    batch_time = utils.meter.AverageMeter()
    losses = utils.meter.AverageMeter()
    top1 = utils.meter.AverageMeter()

    ###############################
    ## confusion table
    ###############################
    confusion = utils.meter.ConfusionMatrix(40)

    # training mode
    model_gnet.eval()

    end = time.time()
    for i, (inputs_12v, labels) in enumerate(test_loader):
        # bz x 12 x 1 x 224 x 224
        # re-view it to be : (bz * 12) x 3 x 224 x 224
        # note that inputs_12v.size(2) = 1, so we expand it to be 3
        inputs_12v = inputs_12v.view(inputs_12v.size(0) * inputs_12v.size(1), inputs_12v.size(2),
                                     inputs_12v.size(3), inputs_12v.size(4))
        labels = labels.long().view(-1)
        if isinstance(inputs_12v, torch.ByteTensor):
            inputs_12v = inputs_12v.float()
        # # expanding: (bz * 12) x 3 x 224 x 224
        #        inputs_12v = inputs_12v.expand(inputs_12v.size(0), 3,
        #            inputs_12v.size(2), inputs_12v.size(3))

        # byte tensor to float tensor
        # normalize data here instead of using clouse in dataset class, because it is
        # not format 12 x 1 x H x W in stead of C x H x W
        #        mean =  223.03979492188
        #        std = 1.0
        #        inputs_12v = utils.preprocess(inputs_12v, mean, std, False) # False means not do data augmentation

        inputs_12v = Variable(inputs_12v, volatile=True)
        labels = Variable(labels, volatile=True)

        # print(points.size())
        # print(labels.size())
        # shift data to GPU
        if opt.cuda:
            inputs_12v = inputs_12v.cuda()
            labels = labels.cuda()  # must be long cuda tensor

        # forward, backward optimize
        # (bz*12) X C x H x W
        preds = model_gnet(inputs_12v)  # bz x C x H x W

        # print(labels)
        # in pytorch, unlike torch, the label is 0-indexed (start from 0)
        #        labels = labels.sub_(1)

        # currently we do not use center loss here
        loss = criterion(preds, labels)

        ###########################################
        ## measure accuracy
        ###########################################
        prec1 = utils.accuracy(preds.data, labels.data, topk=(1,))[0]
        losses.update(loss.data[0], preds.size(0))  # batchsize
        top1.update(prec1[0], preds.size(0))

        ###############################################
        ## confusion table
        ###############################################
        confusion.batchAdd(preds.data, labels.data)

        # debug_here()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1))
            # ###########################################
            # ## Log
            # ###########################################
            # # loss accuracy
            # step = epoch * len(test_loader) + i
            # info = { 'loss' : loss.data[0],
            #         'accuracy': top1.avg}
            #
            # for tag, value in info.items():
            #     logger.scalar_summary(log_pre_name+'test/'+tag, value, step)
            # # parameters gradients
            # for tag, value in model_gnet.named_parameters():
            #     tag = tag.replace('.', '/')
            #     logger.histo_summary(log_pre_name+'test/'+tag, to_np(value), step)
            # # images

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    #####################################
    ## confusion table
    #####################################
    confusion.updateValids()
    print('mean class accuracy at epoch {0}: {1} '.format(epoch, confusion.mean_class_acc))

    # print(tested_samples)
    return top1.avg


def main():
    cfg = utils.config.config()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    train_dataset = datasets.modelnet40.Modelnet40_dataset(cfg, status='train')
    val_dataset = datasets.modelnet40.Modelnet40_dataset(cfg, status='val')

    print('number of train samples is: ', len(train_dataset))
    print('number of test samples is: ', len(val_dataset))

    best_prec1 = 0
    # only used when we resume training from some checkpoint model
    resume_epoch = 0
    # train data loader
    # for loader, droplast by default is set to false
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size,
                                               shuffle=True, num_workers=cfg.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size,
                                              shuffle=True, num_workers=cfg.workers)

    # create model
    model = gvcnn.GVCNN(pretrained=True, aux_logits=cfg.have_aux,
                                   transform_input=True, num_classes=40,
                                   n_views=cfg.data_views, with_group=cfg.with_group)
    if cfg.resume_train:
        print('loading pretrained model from {0}'.format(cfg.ckpt_model))
        checkpoint = torch.load(cfg.ckpt_model)
        model.load_state_dict(checkpoint['model_param_best'])

    print('GVCNN: ')
    print(model)

    # optimizer
    optimizer = optim.SGD(model.parameters(), cfg.lr,
                          momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)

    # if we load model from pretrained, we need the optim state here
    if cfg.resume_train:
        print('loading optim model from {0}'.format(cfg.ckpt_optim))
        optim_state = torch.load(cfg.ckpt_optim)

        resume_epoch = optim_state['epoch']
        best_prec1 = optim_state['best_prec1']
        optimizer.load_state_dict(optim_state['optim_state_best'])

        # cross entropy already contains logsoftmax
    criterion = nn.CrossEntropyLoss()

    if cfg.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        # define loss function (criterion) and pptimizer
        criterion = criterion.cuda()
    # optimizer
    # fix model_prev_pool parameters
    for p in model.parameters():
        p.requires_grad = False  # to avoid computation
    for p in model.fc.parameters():
        p.requires_grad = True  # to  computation
    if cfg.with_group:
        for p in model.fc_q.parameters():
            p.requires_grad = True
    if cfg.have_aux:
        for p in model.AuxLogits.parameters():
            p.requires_grad = True  # to  computation

    for epoch in range(resume_epoch, cfg.max_epoch):
        #################################
        # train for one epoch
        # debug_here()
        #################################
        # debug_here()

        if epoch >= 20:
            for p in model.parameters():
                p.requires_grad = True

        train(train_loader, model, criterion, optimizer, epoch, cfg)

        #################################
        # validate
        #################################
        prec1 = validate(val_loader, model, criterion, optimizer, epoch, opt)

        ##################################
        # save checkpoints
        ##################################
        if best_prec1 < prec1:
            best_prec1 = prec1

            train_helper.save_ckpt(cfg, model, epoch, best_prec1, optim)


        print('best accuracy: ', best_prec1)

if __name__ == '__main__':
    main()

