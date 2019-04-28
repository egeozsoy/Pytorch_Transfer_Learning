from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder


file1 = "cifar-10-batches-py/data_batch_1"
file2 = "cifar-10-batches-py/data_batch_2"
file3 = "cifar-10-batches-py/data_batch_3"
file4 = "cifar-10-batches-py/data_batch_4"
file5 = "cifar-10-batches-py/data_batch_5"
test_file = 'cifar-10-batches-py/test_batch'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def get_data():
    data1 = unpickle(file1)
    data2 = unpickle(file2)
    data3 = unpickle(file3)
    data4 = unpickle(file4)
    data5 = unpickle(file5)
    data = {b'data':np.concatenate((data1[b'data'],data2[b'data'],data3[b'data'],data4[b'data'],data5[b'data'])),
            b'labels':np.concatenate((data1[b'labels'],data2[b'labels'],data3[b'labels'],data4[b'labels'],data5[b'labels']))}
    data[b'data'] = np.transpose(data[b'data'].reshape(-1, 3, 32, 32), (0,2,3,1))
    enc = OneHotEncoder(handle_unknown='ignore')
    X = [[item] for item in data[b'labels']]
    enc.fit(X)
    X = enc.transform(X).toarray()
    data[b'labels'] = X
    return data

def get_test_data():
    data = unpickle(test_file)
    data[b'data'] = np.transpose(data[b'data'].reshape(10000, 3, 32, 32), (0,2,3,1))
    enc = OneHotEncoder(handle_unknown='ignore')
    X = [[item] for item in data[b'labels']]
    enc.fit(X)
    X = enc.transform(X).toarray()
    data[b'labels'] = X
    return data