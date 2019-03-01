"""MNIST MC Dropout Model."""
import cPickle as pickle
import datetime as dt
import gzip
import lasagne as nn
import numpy as np
import os
import sys
import theano
import theano.tensor as T
import time


TITLE = 'mnist_mc_dropout'
OUTPUT_FILE = './output/{}.txt'.format(TITLE)

# Helper functions
from helpers import report, load_mnist_data, generate_batches


def neural_network():
    network = nn.layers.InputLayer(shape=(None, 1, 28, 28))
    network = nn.layers.DropoutLayer(incoming=network, p=.5, rescale=True)
    network = nn.layers.Conv2DLayer(incoming=network, num_filters=32, 
        filter_size=(5, 5), stride=1, nonlinearity=nn.nonlinearities.rectify)
    network = nn.layers.DropoutLayer(incoming=network, p=.5, rescale=True)    
    network = nn.layers.MaxPool2DLayer(incoming=network, pool_size=(2, 2))
    network = nn.layers.DropoutLayer(incoming=network, p=.5, rescale=True)    
    network = nn.layers.Conv2DLayer(incoming=network, num_filters=32, 
        filter_size=(5, 5), stride=1, nonlinearity=nn.nonlinearities.rectify)
    network = nn.layers.DropoutLayer(incoming=network, p=.5, rescale=True)    
    network = nn.layers.MaxPool2DLayer(incoming=network, pool_size=(2, 2))
    network = nn.layers.DropoutLayer(incoming=network, p=.5, rescale=True)
    network = nn.layers.DenseLayer(incoming=network, num_units=500,
        nonlinearity=nn.nonlinearities.rectify)
    network = nn.layers.DropoutLayer(incoming=network, p=.5, rescale=True)    
    network = nn.layers.DenseLayer(incoming=network, num_units=10,
        nonlinearity=nn.nonlinearities.softmax)

    return network


def functions(network):
    # Symbolic variables
    X = T.tensor4()
    Y = T.ivector()

    # Non-deterministic training
    parameters = nn.layers.get_all_params(layer=network, trainable=True)   
    output = nn.layers.get_output(layer_or_layers=network, inputs=X,
        deterministic=False)
    prediction = output.argmax(-1)
    loss = T.mean(nn.objectives.categorical_crossentropy(
        predictions=output, targets=Y))
    accuracy = T.mean(T.eq(prediction, Y))
    gradient = T.grad(cost=loss, wrt=parameters)
    update = nn.updates.nesterov_momentum(loss_or_grads=gradient, 
        params=parameters, learning_rate=0.001, momentum=0.9)
    training_function = theano.function(
        inputs=[X, Y], outputs=[loss, accuracy], updates=update)

    # Non-deterministic testing
    test_function = theano.function(
        inputs=[X], outputs=prediction)

    # Deterministic validation
    det_output = nn.layers.get_output(layer_or_layers=network, inputs=X,
        deterministic=True)
    det_prediction = det_output.argmax(-1)
    det_loss = T.mean(nn.objectives.categorical_crossentropy(
        predictions=det_output, targets=Y))
    det_accuracy = T.mean(T.eq(det_prediction, Y))  
    validation_function = theano.function(
        inputs=[X, Y], outputs=[det_loss, det_accuracy])

    return training_function, validation_function, test_function

if __name__ == '__main__':
    # Record to file instead of printing statements
    report('\n\nStarted: {}.'.format(dt.datetime.now()), OUTPUT_FILE)

    # Obtain data
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()
    report('Data OK.', OUTPUT_FILE)

    # Get network
    network = neural_network()
    report('Network OK.', OUTPUT_FILE)
    report('{} parameters'.format(nn.layers.count_params(network)),OUTPUT_FILE)

    # Get functions
    training_function, validation_function, test_function = functions(network)
    report('Functions OK.', OUTPUT_FILE)

    # Start training
    TL, TA, VL, VA = [], [], [], []
    epochs = 3000
    batch_size = 500
    t_batches = (len(X_train) + (len(X_train) % batch_size)) // batch_size
    v_batches = (len(X_val) + (len(X_val) % batch_size)) // batch_size
    report('Training...', OUTPUT_FILE)
    header = ['Epoch', 'TL', 'TA', 'VL', 'VA', 'Time']
    report('{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}'.format(*header), OUTPUT_FILE)

    for e in xrange(epochs):
        start_time = time.time()
        tl, ta, vl, va = 0., 0., 0., 0.

        # Training round
        for batch in generate_batches(X_train, y_train, batch_size):
            data, targets = batch
            l, a = training_function(data, targets)
            tl += l
            ta += a

        tl /= t_batches
        ta /= t_batches
        TL.append(tl)
        TA.append(ta)

        # Validation round
        for batch in generate_batches(X_val, y_val, batch_size):
            data, targets = batch
            l, a = validation_function(data, targets)
            vl += l
            va += a
        vl /= v_batches
        va /= v_batches
        VL.append(vl)
        VA.append(va)

        row = [e + 1, tl, ta, vl, va, time.time() - start_time]
        report('{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}'.format(*row), OUTPUT_FILE)

    report('Finished training.', OUTPUT_FILE)

    # Save training information
    f = gzip.open('./output/{}_info.pkl.gz'.format(TITLE), 'wb')
    info = {
        'training loss':TL,
        'training accuracy':TA,
        'validation loss':VL,
        'validation accuracy':VA
    }
    pickle.dump(info, f)
    f.close()
    report('Saved training info.', OUTPUT_FILE)

    # Check model's accuracy on test data
    test_loss, test_acc = validation_function(X_test, y_test)
    report('Loss on test data: {}'.format(test_loss), OUTPUT_FILE)
    report('Accuracy on test data: {}%'.format(test_acc * 100), OUTPUT_FILE)

    # Save weights
    weights = nn.layers.get_all_params(network)
    weights = [np.array(w.get_value()) for w in weights]
    f = gzip.open('./output/{}_weights.pkl.gz'.format(TITLE), 'wb')
    pickle.dump(weights, f)
    f.close()
    report('Saved weights.', OUTPUT_FILE)

    # Done
    report('Completed: {}.'.format(dt.datetime.now()), OUTPUT_FILE)