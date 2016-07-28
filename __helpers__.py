import os
import gzip
from urllib import urlretrieve
import cPickle as pickle
import numpy as np

""" helper functions """

def report(text, output_file):
    f = open(output_file, 'a')
    f.write('{}\n'.format(text))
    f.close


def load_mnist_data():
    mnist_filename = 'mnist.pkl.gz'
    if not os.path.exists(mnist_filename):
        print 'Downloading MNIST data ...'
        url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        urlretrieve(url, mnist_filename)
    train, val, test = pickle.load(gzip.open(mnist_filename, 'rb'))
    # training dataset
    X_train, Y_train = train
    X_train = X_train.reshape((-1, 1, 28, 28)).astype('float32')
    Y_train = Y_train.astype('int32')
    # validation dataset
    X_val, Y_val = val
    X_val = X_val.reshape((-1, 1, 28, 28)).astype('float32')
    Y_val = Y_val.astype('int32')
    # test dataset
    X_test, Y_test = test
    X_test = X_test.reshape((-1, 1, 28, 28)).astype('float32')
    Y_test = Y_test.astype('int32')
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def generate_batches(data, target, batch_size=500, stochastic=True):
    idx = np.arange(len(data))
    np.random.shuffle(idx) if stochastic else idx
    for k in xrange(0, len(data), batch_size):
        sample = idx[slice(k, k + batch_size)]
        yield data[sample], target[sample]

