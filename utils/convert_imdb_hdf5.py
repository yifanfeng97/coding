# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 20:33:01 2017

@author: Administrator
"""

import h5py
import scipy.io as scio
import os
from PIL import Image
import numpy as np
#import flags as FLAGS

#VGG_MEAN =[103.939, 116.779, 123.68]

def process(imrgb, convert = False):
    data = np.array(imrgb) 
#    red, green, blue = data.T.astype(np.float32)
#    data = np.array([blue - VGG_MEAN[0], green- VGG_MEAN[1], red- VGG_MEAN[2]])
    red, green, blue = data.transpose()
    if convert:
        data = np.array([blue, green, red])
    else:
        data = np.array([red, green, blue])
#    data[data==255]=0
#    data = data.transpose()
#    imbgr = Image.fromarray(data.astype(np.int8))
    data = data.astype(np.uint8)
    return data

n_views = 12
img_root = r'../data/%d_ModelNet40'%n_views
#img_root = r'modelnet40v1'
h5_root = r'../data/data_h5'
# imsize = 299
imsize = 224
imdb_dir = os.path.join(img_root, 'imdb.mat')
#view_num = FLAGS.view_num

imdb = scio.loadmat(imdb_dir)

f = h5py.File(os.path.join(h5_root, '%d_modelnet40_%d.h5'%(n_views, imsize)), 'w')

idx_name={'train':1, 'validation':2, 'test':3}
names = ['train', 'validation', 'test']
f.create_group('train')
f.create_group('test')
f.create_group('validation')
for name in f:
    no = idx_name[name]
    num = int((imdb['images']['set'][0][0][0]==no).sum()/n_views)
    print('%s:%d'%(name, num))
    f[name].create_dataset('data', shape=(num, n_views, 3, imsize, imsize), dtype='uint8')
    f[name].create_dataset('label', shape=(num, 1), dtype = 'uint8')

idx_num=[0, 0, 0]
num_all = imdb['images']['id'][0][0][0][-1]
for i in range(0, num_all, n_views):
    no = imdb['images']['set'][0][0][0][i]-1
    name = names[no]
    print('%d/%d'%(i, imdb['images']['id'][0][0][0][-1]))
    f[name]['label'][idx_num[no]] = imdb['images']['class'][0][0][0][i] - 1
    print(f[name]['label'][idx_num[no]])
    for tmp in range(n_views):
        imdir = os.path.join(img_root, (imdb['images']['name'][0][0][0][i+tmp][0]).replace('\\','/'))
        im = Image.open(imdir).resize((imsize, imsize))
        #convert RGB to BGR
        im = process(im)
        f[name]['data'][idx_num[no], tmp, ...] = im.astype(np.uint8) 
    idx_num[no]+=1
f.close()

    
    


