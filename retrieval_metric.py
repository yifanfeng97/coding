import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import config

# all 2468 shapes
data_dir = r'../data'
feature_dir = 'features'
top_k = 1000


def des_fun(a, b):
    return np.linalg.norm(a - b)


def cal_des(model_views = 8, input_views = 8, save = True, with_Metric = False):
    if with_Metric:
        ft_dir = os.path.join(data_dir, feature_dir, '%dm_%dv_Metric_modelnet40.h5' % (model_views, input_views))
    else:
        ft_dir = os.path.join(data_dir, feature_dir, '%dm_%dv_modelnet40.h5' % (model_views, input_views))
    fft = h5py.File(ft_dir, 'r')
    features = fft['test']['features']
    num = len(features)
    des_mat = np.zeros((num, num), dtype=np.float32)
    for i in range(num):
        for j in range(i, num):
            tmp = des_fun(features[i], features[j])
            des_mat[i][j] = tmp
            des_mat[j][i] = tmp
            print('%d/%d\t%.1f%%'%(i*num+j+1, num*(num-1), (i*num+j+1)/(num*(num-1))*100))

    if save == True:
        if with_Metric:
            ft_dir = os.path.join(data_dir, feature_dir, '%dm_%dv_Metric_des_modelnet40.h5' % (model_views, input_views))
        else:
            ft_dir = os.path.join(data_dir, feature_dir, '%dm_%dv_des_modelnet40.h5' % (model_views, input_views))
        fdes = h5py.File(ft_dir, 'w')
        fdes.create_dataset('destance', shape=(num, num), data=des_mat)
        fdes.create_dataset('labels', data=fft['test']['labels'])
        fdes.close()
    fft.close()
    read_cal_map(model_views = model_views, input_views = input_views, with_Metric = with_Metric)

def read_cal_map(model_views = 8, input_views = 8, with_Metric = False):
    if with_Metric:
        ft_dir = os.path.join(data_dir, feature_dir, '%dm_%dv_Metric_des_modelnet40.h5' % (model_views, input_views))
    else:
        ft_dir = os.path.join(data_dir, feature_dir, '%dm_%dv_des_modelnet40.h5' % (model_views, input_views))
    fdes = h5py.File(ft_dir, 'r')
    labels = fdes['labels'][:]
    des_mat = fdes['destance'][:]
    num = len(labels)
    mAP = 0
    for i in range(num):
        scores = des_mat[:, i]
        targets = (labels == labels[i]).astype(np.uint8)
        sortind = np.argsort(scores, 0)[:top_k]
        truth = targets[sortind]
        sum = 0
        precision = []
        for j in range(top_k):
            if truth[j]:
                sum+=1
                precision.append(sum/(j + 1))
        if len(precision) == 0:
            ap = 0
        else:
            for i in range(len(precision)):
                precision[i] = max(precision[i:])
            ap = np.array(precision).mean()
        mAP += ap
        print('%d/%d\tap:%f'%(i+1, num, ap), precision)
    print(mAP/num)

def read_cal_pr(model_views = 8, input_views = 8, with_Metric = False, save = True, draw = False):
    if with_Metric:
        ft_dir = os.path.join(data_dir, feature_dir, '%dm_%dv_Metric_des_modelnet40.h5' % (model_views, input_views))
    else:
        ft_dir = os.path.join(data_dir, feature_dir, '%dm_%dv_des_modelnet40.h5' % (model_views, input_views))
    fdes = h5py.File(ft_dir, 'r')
    labels = fdes['labels'][:]
    des_mat = fdes['destance'][:]
    num = len(labels)
    precisions = []
    recalls = []
    ans = []
    for i in range(num):
        scores = des_mat[:, i]
        targets = (labels == labels[i]).astype(np.uint8)
        sortind = np.argsort(scores, 0)[:top_k]
        truth = targets[sortind]
        tmp = 0
        sum = truth[:top_k].sum()
        precision = []
        recall = []
        for j in range(top_k):
            if truth[j]:
                tmp+=1
                # precision.append(sum/(j + 1))
            recall.append(tmp*1.0/sum)
            precision.append(tmp*1.0/(j+1))
        precisions.append(precision)
        for j in range(len(precision)):
            precision[j] = max(precision[j:])
        recalls.append(recall)
        #
        tmp = []
        for ii in range(11):
            min_des = 100
            val = 0
            for j in range(top_k):
                if abs(recall[j] - ii * 0.1) < min_des:
                    min_des = abs(recall[j] - ii * 0.1)
                    val = precision[j]
            tmp.append(val)
        print('%d/%d'%(i+1, num))
        ans.append(tmp)
    # precisions = np.array(precisions).mean(0)
    # recalls = np.array(recalls).mean(0)
    # for i in range(11):
    #     min_des = 100
    #     val = 0
    #     for j in range(top_k):
    #         if abs(recalls[j] - i*0.1) < min_des:
    #             min_des = abs(recalls[j] - i*0.1)
    #             val = precisions[j]
    #     ans.append(val)
    ans = np.array(ans).mean(0)
    if save:
        if with_Metric:
            save_dir = os.path.join(data_dir, 'paper_data', 'pr_%dm_%dv_Metric_modelnet40.csv' % (model_views, input_views))
        else:
            save_dir = os.path.join(data_dir, 'paper_data', 'pr_%dm_%dv_modelnet40.csv' % (model_views, input_views))
        np.savetxt(save_dir, np.array(ans), fmt='%.3f', delimiter=',')
    if draw:
        plt.plot(ans)
        plt.show()

def test():
    scores = [0.23, 0.76, 0.01, 0.91, 0.13, 0.45, 0.12, 0.03, 0.38, 0.11, 0.03, 0.09, 0.65, 0.07, 0.12, 0.24, 0.1, 0.23, 0.46, 0.08]
    gt_label = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    scores = np.array(scores)
    targets = np.array(gt_label).astype(np.uint8)
    sortind = np.argsort(scores, 0)[::-1]
    truth = targets[sortind]
    sum = 0
    precision = []
    for j in range(20):
        if truth[j]:
            sum += 1
            precision.append(sum / (j + 1))
    if len(precision) == 0:
        ap = 0
    else:
        for i in range(len(precision)):
            precision[i] = max(precision[i:])
        ap = np.array(precision).mean()
    print(ap)


if __name__ == '__main__':
    draw = True
    # cal_des(model_views=12, input_views=12, save=True)
    # cal_des(model_views=12, input_views=12, save=True, with_Metric=True)
    # read_cal_map(model_views=8, input_views=8, with_Metric=True)
    # read_cal_map(model_views=8, input_views=8)
    # read_cal_pr(model_views=8, input_views=8, with_Metric=True, draw = draw)
    # read_cal_pr(model_views=8, input_views=8, with_Metric=False, draw = draw)
    # read_cal_pr(model_views=12, input_views=12, with_Metric=True, draw = draw)
    read_cal_pr(model_views=12, input_views=12, with_Metric=False, draw = draw)
    # test()
