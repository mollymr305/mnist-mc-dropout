"""Visualize uncertainty."""
import collections
import cPickle as pickle
import gzip
import lasagne as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import theano
import theano.tensor as T


from helpers import load_mnist_data
from mnist_mc_dropout import neural_network, functions

def main(idx, net, data, target, test_function, T=100):
    sample = [test_function([data[idx]])[0] for _ in range(T)]
    print "\n\nIndex = {}.".format(idx)
    print np.array(sample)

    model_answer = collections.Counter(sample).most_common(1)
    print "Model answer = {}.".format(model_answer)
    print "Correct answer = {}.".format(target[idx])

    height = [sample.count(i) / float(T) for i in range(10)]
    left = [i for i in range(10)]
    tick_label = [str(i) for i in range(10)]

    plt.switch_backend('Agg')
    plt.subplot(1, 2, 1)
    plt.bar(left=left, height=height, align='center', tick_label=tick_label)
    plt.tick_params(axis='y', which='major', labelsize=15)
    plt.tick_params(axis='x', which='major', labelsize=20)
    plt.subplot(1, 2, 2)
    plt.imshow(data[idx].reshape(28,28))
    plt.title('{}\n'.format(y_test[idx]), size=22)
    plt.axis('off')
    plt.savefig('./output/index_{}.eps'.format(idx),format='eps')
    plt.savefig('./output/index_{}.jpg'.format(idx),format='jpg')
    plt.close()

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()
    net = neural_network()
    weights = pickle.load(gzip.open('./output/mnist_mc_dropout_weights.pkl.gz', 'rb'))
    nn.layers.set_all_param_values(net, weights)
    training_function, validation_function, test_function = functions(net)

    for idx in range(100):
        main(idx, net, X_test, y_test, test_function)