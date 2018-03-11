import utils.config
import pickle

def random_split():
    cfg = utils.config.config()
    train = []
    test = []
    train_data = []
    val_data = []
    train = get_file_list('train', cfg.data_root, cfg.data_views)


def get_file_list(data_type, root, data_views):
    pass


if __name__ == '__main__':
    random_split()