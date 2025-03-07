import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp
import sys
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）


sys.path.append('/root/autodl-tmp/HGB_autobenchmark/NC/benchmark/scripts/')
sys.path.append('/root/autodl-tmp/HGB_autobenchmark/NC/benchmark/methods/GNN/utils/')

def load_data(prefix='DBLP'):
    import data_loader
    # from scripts.data_loader import data_loader

    # from scripts.data_loader import some_function

    print(prefix)
    dl = data_loader.data_loader('/root/autodl-tmp/HGB_autobenchmark/data/'+prefix)
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    if prefix != 'IMDB':
        labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    emb = np.load('/root/autodl-tmp/HGB_autobenchmark/data/'+ prefix + '/metapath2vec_emb_node.npy')
    return features,\
           adjM, \
           labels,\
           train_val_test_idx,\
            dl,emb
