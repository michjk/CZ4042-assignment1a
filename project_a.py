import numpy as np
import os
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import time

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-np.min(X, axis=0))

# load dataset
def load_data(data_path):
    input_txt = np.loadtxt('dataset/sat_train.txt',delimiter=' ')
    X, _Y = input_txt[:,:36], input_txt[:,-1].astype(int)
    X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
    X = scale(X, X_min, X_max)

    _Y[_Y == 7] = 6
    Y = np.zeros((_Y.shape[0], 6))
    Y[np.arange(_Y.shape[0]), _Y-1] = 1

    return X, Y

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

def create_3_layer_NN(decay, learning_rate, n_input, n_hidden, n_output):
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

    cost = T.mean(T.nnet.categorical_crossentropy(py, Y)) + decay*(T.sum(T.sqr(w1)+T.sum(T.sqr(w2))))
    params = [w1, b1, w2, b2]
    updates = sgd(cost, params, learning_rate)

    # compile
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

    return train, predict


# suffle data
def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

def run_NN_model(train, predict, batch_size, trainX, trainY, testX, testY, epochs):
    # train and test
    n = len(trainX)
    test_accuracy = []
    train_cost = []
    for i in range(epochs):
        if i%100 == 0:
            print("Epoch: %d"%i)
        
        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0.0
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            cost += train(trainX[start:end], trainY[start:end])
        train_cost = np.append(train_cost, cost/(n // batch_size))
        
        test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))

    print('%.1f accuracy at %d iterations'%(np.max(test_accuracy)*100, np.argmax(test_accuracy)+1))

    return train_cost, test_accuracy

def createNewFolder(folder_path):
    #Create folder for saving figure
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

def draw_plot(listX, listY, labelX, labelY, title, save_folder_path, file_name):
    createNewFolder(save_folder_path)

    plt.figure()
    plt.plot(listX, listY)
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.title(title)
    plt.savefig(os.path.join(save_folder_path, file_name))
    plt.show()

def draw_multi_plot(listX, dict_listY, labelX, labelY, list_plot_label, prefix_plot_label, title, save_folder_path, file_name):
    createNewFolder(save_folder_path)

    plt.figure()

    for plot_label in list_plot_label:
        plt.plot(listX, dict_listY[plot_label], label=str(prefix_plot_label)+str(plot_label))
    
    plt.legend()
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.title(title)
    plt.savefig(os.path.join(save_folder_path, file_name))
    plt.show()

decay = 1e-6
learning_rate = 0.01
n_input = 36
n_hidden = 10
n_output = 6
epochs = 1000
batch_size = 32
dataset_dir_path = "dataset"
train_data_path = os.path.join(dataset_dir_path, "sat_train.txt")
test_data_path = os.path.join(dataset_dir_path, "sat_test.txt")
figure_dir_path = "figure"

#read train data
trainX, trainY = load_data(train_data_path)

#read test data
testX, testY = load_data(test_data_path)

print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)

#create train and predict function
train, predict = create_3_layer_NN(decay, learning_rate, n_input, n_hidden, n_output)

# train and test
train_cost, test_accuracy = run_NN_model(train, predict, batch_size, trainX, trainY, testX, testY, epochs)

#Plots
draw_plot(range(epochs), train_cost, 'iteration', 'cross-entropy', 'training cost', figure_dir_path, 'pa1_cost.png')
draw_plot(range(epochs), test_accuracy, 'iteration', 'accuracy', 'test accuracy', figure_dir_path, 'pa1_accuracy.png')

batch_sizes = [4,8,16,32,64]

# Question 2
batch_sizes = [4,8,16,32,64]
time_for_update = np.zeros(max(batch_sizes) + 1)
train_cost_all_batch = {}
test_accuracy_all_batch = {}

for batch_size in batch_sizes:
    print("Batch size %d"%(batch_size))
    # create model
    train, predict = create_3_layer_NN(decay, learning_rate, n_input, n_hidden, n_output)
    
    # run model
    t = time.time()
    train_cost, test_accuracy = run_NN_model(train, predict, batch_size, trainX, trainY, testX, testY, epochs)

    time_for_update[batch_size] = 1000*(time.time()-t)/epochs
    train_cost_all_batch[batch_size] = train_cost
    test_accuracy_all_batch[batch_size] = test_accuracy

# plots
draw_multi_plot(range(epochs), train_cost_all_batch, 'iteration', 'cross-entropy', batch_sizes, "b=", 'training cost', figure_dir_path, 'pa2_cost.png')
draw_multi_plot(range(epochs), test_accuracy_all_batch, 'iteration', 'cross-entropy', batch_sizes, "b=", 'test accuracy', figure_dir_path, 'pa2_accuracy.png')
draw_plot(batch_sizes, time_for_update[batch_sizes], 'batch size', 'time for update in ms', 'time for a weight update', figure_dir_path, 'pa2_time_update.png')


