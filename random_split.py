import utils.config
import os
import os.path as osp
import pickle
import glob

def random_split():
    cfg = utils.config.config()
    train = []
    test = []
    train_data = []
    val_data = []
    train = get_filename_list('train', cfg.data_root, cfg.data_views)

def get_d_list(d_root, data_views):
    pass


def get_filename_list(data_type, root, data_views):
    filename_list = []
    data_all = glob.glob(osp.join(root, '*'))
    data_all = [data for data in data_all if osp.isdir(data)]
    for _idx, d in enumerate(data_all):
        d_lbl = osp.split(d)[1]
        d_lbl_idx = _idx
        d_root = osp.join(root, data_type)
        d_list = get_d_list(d_root, data_views)
        filename_list.append({'label': d_lbl,
                              'label_idx': d_lbl_idx,
                              'imgs': d_list})
    return filename_list

    print(data_all)


if __name__ == '__main__':
    random_split()