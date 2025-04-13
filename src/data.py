import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_cifar_batch(filename):
    """加载单个 CIFAR-10 数据文件"""
    with open(filename, 'rb') as f:
        dict_data = pickle.load(f, encoding='bytes')
        X = dict_data[b'data']  # shape (10000, 3072)
        y = dict_data[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  # (N,32,32,3)
        y = np.array(y)
    return X, y

def load_cifar10(root):
    """加载整个 CIFAR-10 数据集"""
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % b)
        X, y = load_cifar_batch(f)
        xs.append(X)
        ys.append(y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    X_test, y_test = load_cifar_batch(os.path.join(root, 'test_batch'))
    return (X_train, y_train), (X_test, y_test)

def preprocess_data(X):
    """将图像数据展平，并归一化到 [0,1] 之间"""
    X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
    return X_flat

def shuffle_data(X, y):
    """打乱数据顺序"""
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]
