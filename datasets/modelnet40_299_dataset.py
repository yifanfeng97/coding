from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import errno
import torch
import json
import h5py

import numpy as np
import sys

import json


class Modelnet40_Dataset(data.Dataset):
    def __init__(self, data_dir, image_size = 299, train=True, n_views = 12):
        self.image_size = image_size
        self.data_dir = data_dir
        self.train = train 


        file_path = os.path.join(self.data_dir, '%d_modelnet40_299.h5'%n_views)
        self.modelnet40_data = h5py.File(file_path)
        
        if self.train: 
            # when loading data from numpy array, we must convert data from numpy, otherwise, it will cause error
            # below will load all the data into memory 
            self.train_data = torch.from_numpy(self.modelnet40_data['train']['data'].value)
            self.train_labels = torch.from_numpy(self.modelnet40_data['train']['label'].value)
        else:
            self.test_data = torch.from_numpy(self.modelnet40_data['test']['data'].value)
            self.test_labels = torch.from_numpy(self.modelnet40_data['test']['label'].value)

    def __getitem__(self, index):
        if self.train:
            shape, label = self.train_data[index], self.train_labels[index]
            # shape_12v, label = self.train_data['train']['data'][index], self.train_data['train']['label'][index]
        else:
            shape, label = self.test_data[index], self.test_labels[index]
        return shape, label 

    
    # method 1 
    def compute_mean_std(self):
        assert self.train=='train', 'must compute mean of training dataset'
        self.mean = 0
        self.std = 0
        # modelnet12v is grey scale images
        for k in range(self.train_data.size(0)): 
            # load sample
            sample = self.train_data[k] # bytetensor, ant bytetensor has not mean methods
            self.mean = self.mean + sample.float().mean()
            self.std = self.std + sample.float().std()  
        # compute mean of the dataset 
        self.mean = self.mean/self.train_data.size(0)
        self.std = self.std/self.train_data.size(0)


        print("mean: {0}".format(self.mean))
        print("std: {0}".format(self.std))
    # mean 2 
    def compute_mean_std2(self):
        assert self.train=='train', 'must compute mean of training dataset'
        self.mean2 = 0
        self.std2 = 0
        # modelnet12v is grey scale images
        # reading all dataset 
        whole_train_dataset = self.train_data[:].float() 

        # compute mean of the dataset 
        self.mean2 = whole_train_dataset.mean() 
        self.std2 = whole_train_dataset.std() 

        print("mean: {0}".format(self.mean2))
        print("std: {0}".format(self.std2))


    def __len__(self):
        if self.train:
            return self.train_data.size(0)
            # return self.train_data['train']['data'].shape[0]
        else:
            return self.test_data.size(0)


class Modelnet40_r20_Dataset(data.Dataset):
    def __init__(self, data_dir, image_size=299):
        self.image_size = image_size
        self.data_dir = data_dir

        file_12_path = os.path.join(self.data_dir, '12_modelnet40_299.h5')
        file_8_path = os.path.join(self.data_dir, '8_modelnet40_299.h5')
        self.modelnet40_12_data = h5py.File(file_12_path)
        self.modelnet40_8_data = h5py.File(file_8_path)


        self.data_8 = torch.from_numpy(self.modelnet40_8_data['test']['data'].value)
        self.labels_8 = torch.from_numpy(self.modelnet40_8_data['test']['label'].value)

        self.data_12 = torch.from_numpy(self.modelnet40_12_data['test']['data'].value)
        self.labels_12 = torch.from_numpy(self.modelnet40_12_data['test']['label'].value)

    def __getitem__(self, index):
        assert self.labels_8[index].numpy() == self.labels_12[index].numpy()
        # shape_8, label = self.data[index], self.labels[index]
        return self.data_8[index], self.data_12[index], self.labels_8[index]


    def __len__(self):
        return self.data_8.size(0)

if __name__ == '__main__':
    print('test')
    train_dataset = Modelnet40_Dataset(data_dir='/home/hxw/project_work_on/shape_research/multiview_cnn/data', train=True)
    # train_dataset.compute_mean_std()
    print(len(train_dataset))

    test_dataset = Modelnet40_Dataset(data_dir='/home/hxw/project_work_on/shape_research/multiview_cnn/data', train=False)
    print(len(test_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8,
                                     shuffle=True, num_workers=2)


    total = 0
    # check when to cause labels error 
    for epoch in range(200):
        print('epoch', epoch)
        for i, (input_v, labels) in enumerate(train_loader):
            total = total + labels.size(0)

            # labels can be 255, what is the problem??
            if labels.max() > 40:
                print('error')

            if labels.min() < 1:
                print('error')
            
            labels.sub_(1) # minus 1 in place 
            
            if labels.max() >= 40:
                print('error')

            if labels.min() < 0:
                print('error')
        print(total)


