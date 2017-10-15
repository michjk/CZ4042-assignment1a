import numpy as np
import os
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import time

# create bias tensor
def init_bias(n = 1):
    return(theano.shared(np.zeros(n), theano.config.floatX))

# create weights tensor
def init_weights(n_in=1, n_out=1, logistic=True):
    W_values = np.asarray(
        np.random.uniform(
        low=-np.sqrt(6. / (n_in + n_out)),
        high=np.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)),
        dtype=theano.config.floatX
        )
    if logistic == True:
        W_values *= 4
    return (theano.shared(value=W_values, name='W', borrow=True))

# update parameters
def sgd(cost, params, lr=0.01):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def create_3_layer_NN(decay, learning_rate, n_input, n_hidden, n_output, debug_parameter = False):
    #Info
    if debug_parameter:
        print("## Create NN ##")
        print("Decay: " + str(decay))
        print("Learning rate: " + str(learning_rate))
        print("Input size: %d"%n_input)
        print("Hidden size: %d"%n_hidden)
        print("Output size: %d\n"%n_output)

    # theano expressions
    X = T.matrix() #features
    Y = T.matrix() #output

    w1, b1 = init_weights(n_input, n_hidden), init_bias(n_hidden) #weights and biases from input to hidden layer
    w2, b2 = init_weights(n_hidden, n_output, logistic=False), init_bias(n_output) #weights and biases from hidden to output layer
    
    # connect layer
    h1 = T.nnet.sigmoid(T.dot(X, w1) + b1)
    py = T.nnet.softmax(T.dot(h1, w2) + b2)

    # decode to category number
    y_x = T.argmax(py, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(py, Y)) + decay*(T.sum(T.sqr(w1))+T.sum(T.sqr(w2)))
    params = [w1, b1, w2, b2]
    updates = sgd(cost, params, learning_rate)

    # compile
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

    return train, predict

def create_4_layer_NN(decay, learning_rate, n_input, n_hidden1, n_hidden2, n_output, debug_parameter = True):
    #debug
    if debug_parameter:
        print("## Create NN ##")
        print("Decay: " + str(decay))
        print("Learning rate: " + str(learning_rate))
        print("Input size: %d"%n_input)
        print("Hidden1 size: %d"%n_hidden1)
        print("Hidden2 size: %d"%n_hidden2)
        print("Output size: %d\n"%n_output)
    
    # theano expressions
    X = T.matrix() #features
    Y = T.matrix() #output

    w1, b1 = init_weights(n_input, n_hidden1), init_bias(n_hidden1) #weights and biases from input to hidden layer
    w2, b2 = init_weights(n_hidden1, n_hidden2), init_bias(n_hidden2) #weights and biases from hidden 1 to hidden 2 layer
    w3, b3 = init_weights(n_hidden2, n_output, logistic=False), init_bias(n_output) #weights and biases from hidden 2 to output layer

    # connect layer
    h1 = T.nnet.sigmoid(T.dot(X, w1) + b1)
    h2 = T.nnet.sigmoid(T.dot(h1, w2) + b2)
    py = T.nnet.softmax(T.dot(h2, w3) + b3)

    # decode to category number
    y_x = T.argmax(py, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(py, Y)) + decay*(T.sum(T.sqr(w1))+T.sum(T.sqr(w2)) + T.sum(T.sqr(w3)))
    params = [w1, b1, w2, b2, w3, b3]
    updates = sgd(cost, params, learning_rate)
    
    # compile
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=[y_x], allow_input_downcast=True)

    return train, predict


# suffle data
def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

def run_NN_model(train, predict, batch_size, trainX, trainY, testX, testY, epochs, debug_iteration = False):
    #debug
    print("## Run NN with batch size: %d and epochs: %d ##"%(batch_size, epochs))
    # train and test
    n = len(trainX)
    test_accuracy = []
    train_cost = []
    for i in range(epochs):
        if debug_iteration and i%100 == 0:
            print("Epoch: %d"%i)
        
        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0.0
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            cost += train(trainX[start:end], trainY[start:end])
        train_cost = np.append(train_cost, cost/(n // batch_size))
        
        test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))

    print('%.1f accuracy at %d iterations'%(np.max(test_accuracy)*100, np.argmax(test_accuracy)+1))

    return train_cost, test_accuracy