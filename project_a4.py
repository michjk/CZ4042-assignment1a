import numpy as np
import os
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import time
import data_preprocessor as dp 
import model 
import data_visualizer as dv

np.random.seed(0)

learning_rate = 0.01
n_input = 36
n_hidden = 15
n_output = 6
batch_size = 16
epochs = 1000
list_decay = [1e-3, 1e-6, 1e-9, 1e-12, 0]

dataset_dir_path = "dataset"
train_data_path = os.path.join(dataset_dir_path, "sat_train.txt")
test_data_path = os.path.join(dataset_dir_path, "sat_test.txt")
figure_dir_path = "figure"

#read train data
trainX, trainY, X_min, X_max = dp.load_data(train_data_path)

#read test data
testX, testY, _, _ = dp.load_data(test_data_path, True, X_min, X_max)

train_cost_all_decay = {}
test_accuracy_all_decay = {}
train_size = len(trainX)

for decay in list_decay:
    # create model
    train, predict = model.create_3_layer_NN(decay, learning_rate, n_input, n_hidden, n_output, debug_parameter=True)
    
    # run model
    train_cost, test_accuracy = model.run_NN_model(train, predict, batch_size, trainX, trainY, testX, testY, epochs)

    train_cost_all_decay[decay] = train_cost
    test_accuracy_all_decay[decay] = test_accuracy

#Plots
dv.draw_multi_plot(range(epochs), train_cost_all_decay, 'iteration', 'cross-entropy', list_decay, "d=", 'training cost', figure_dir_path, 'pa4_cost.png')
dv.draw_multi_plot(range(epochs), test_accuracy_all_decay, 'iteration', 'cross-entropy', list_decay, "d=", 'test accuracy', figure_dir_path, 'pa4_accuracy.png')
