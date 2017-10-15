import numpy as np
import os
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import time
import data_preprocessor as dp 
import model 
import data_visualizer as dv

#set seed
np.random.seed(0)

decay = 0
learning_rate = 0.01
n_input = 36
batch_size = 16
n_output = 6
epochs = 1000

dataset_dir_path = "dataset"
train_data_path = os.path.join(dataset_dir_path, "sat_train.txt")
test_data_path = os.path.join(dataset_dir_path, "sat_test.txt")
figure_dir_path = "figure"

#read train data
trainX, trainY, X_min, X_max = dp.load_data(train_data_path)

#read test data
testX, testY, _, _ = dp.load_data(test_data_path, True, X_min, X_max)

hidden_sizes = [5, 10, 15, 20, 25]
time_for_update = np.zeros(max(hidden_sizes) + 1)
train_cost_all_hidden = {}
test_accuracy_all_hidden = {}
train_size = len(trainX)

for n_hidden in hidden_sizes:
    # create model
    train, predict = model.create_3_layer_NN(decay, learning_rate, n_input, n_hidden, n_output, debug_parameter = True)
    
    # run model
    t = time.time()
    train_cost, test_accuracy = model.run_NN_model(train, predict, batch_size, trainX, trainY, testX, testY, epochs)

    time_for_update[n_hidden] = 1000*(time.time()-t)/epochs
    train_cost_all_hidden[n_hidden] = train_cost
    test_accuracy_all_hidden[n_hidden] = test_accuracy

#Plots
dv.draw_multi_plot(range(epochs), train_cost_all_hidden, 'iteration', 'cross-entropy', hidden_sizes, "h=", 'training cost', figure_dir_path, 'pa3_cost.png')
dv.draw_multi_plot(range(epochs), test_accuracy_all_hidden, 'iteration', 'cross-entropy', hidden_sizes, "h=", 'test accuracy', figure_dir_path, 'pa3_accuracy.png')
dv.draw_plot(hidden_sizes, time_for_update[hidden_sizes], 'hidden size', 'time for update in ms', 'time for a weight update', figure_dir_path, 'pa3_time_update.png')