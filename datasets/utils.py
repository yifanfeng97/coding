import torch 
import torch.nn as nn 
import numpy as np
import os
import time
from torch.autograd import Variable

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n # val*n: how many samples predicted correctly among the n samples 
        self.count += n     # totoal samples has been through 
        self.avg = self.sum / self.count 

#################################################
## confusion matrix 
#################################################
class ConfusionMatrix(object):
    def __init__(self, K): # K is number of classes 
        self.reset(K) 
    def reset(self, K):
        self.num_classes = K 
        # declare a table matrix and zero it 
        self.cm = torch.zeros(K, K) # one row for each class, column is predicted class 
        # self.valids
        self.valids = torch.zeros(K) 
        # mean average precision, i.e., mean class accuracy  
        self.mean_class_acc = 0 

    def batchAdd(self, outputs, targets): 
        """
        output is predicetd probability 
        """
        _, preds = outputs.topk(1, 1, True, True)
        # convert cudalong tensor to long tensor 
        # preds:  bz x 1 
#        print(outputs.size(), targets.size())
#        print(outputs)
#        print(preds)
        for m in range(preds.size(0)):
#            print('targets')
#            print(targets[m])
#            print('pred')
#            print(preds[m][0])
            self.cm[targets[m]][preds[m][0]] = self.cm[targets[m]][preds[m][0]] + 1 

    def updateValids(self):
        # total = 0 
        for t in range(self.num_classes):
            if self.cm.select(0, t).sum() != 0: # column  
                # sum of t-th row is the number of samples coresponding to this class (groundtruth)
                self.valids[t] = self.cm[t][t] / self.cm.select(0, t).sum()
            else:
                self.valids[t] = 0 

        self.mean_class_acc = self.valids.mean() 

        
    


#################################################
## compute accuracy 
#################################################
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # top k 
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(model, output_path):    

    ## if not os.path.exists(output_dir):
    ##    os.makedirs("model/")        
    torch.save(model, output_path)
        
    print("Checkpoint saved to {}".format(output_path))


# do gradient clip 
def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            param.grad.data.clamp_(-grad_clip, grad_clip)


def preprocess(inputs_12v, mean, std, data_augment):
    """
    inputs_12v: (bz * 12) x 3 x 224 x 224 
    """
    # to tensor 
    if isinstance(inputs_12v, torch.ByteTensor):
        inputs_12v = inputs_12v.float() 

    inputs_12v.sub_(mean).div_(std)

    if data_augment: 
        print('currently not support data augmentation')

    return inputs_12v


def get_center_loss(centers, features, target, alpha, num_classes):
    batch_size = target.size(0)
    features_dim = features.size(1)
    
    target_expand = target.view(batch_size,1).expand(batch_size,features_dim)
    centers_var = Variable(centers)
    centers_batch = centers_var.gather(0,target_expand)
    criterion = nn.MSELoss()
    center_loss = criterion(features,  centers_batch)
    
    # compute gradient w.r.t. center
    diff = centers_batch - features
    unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
    appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
    appear_times_expand = appear_times.view(-1,1).expand(batch_size,features_dim).type(torch.FloatTensor)
    diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
    diff_cpu = alpha * diff_cpu

    # update related centers 
    for i in range(batch_size):
        centers[target.data[i]] -= diff_cpu[i].type(centers.type())

    return center_loss, centers


def get_contrastive_center_loss(centers, beta):
    center_dim = centers.size(1)
    
    target_expand = target.view(batch_size,1).expand(batch_size,features_dim)
    centers_var = Variable(centers)
    centers_batch = centers_var.gather(0,target_expand)
    criterion = nn.MSELoss()
    center_loss = criterion(features,  centers_batch)
    
    # compute gradient w.r.t. center
    diff = centers_batch - features
    unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
    appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
    appear_times_expand = appear_times.view(-1,1).expand(batch_size,features_dim).type(torch.FloatTensor)
    diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
    diff_cpu = alpha * diff_cpu

    # update related centers 
    for i in range(batch_size):
        centers[target.data[i]] -= diff_cpu[i].type(centers.type())

    return center_loss, centers










