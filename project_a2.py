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

decay = 1e-6
learning_rate = 0.01
n_input = 36
n_hidden = 10
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

batch_sizes = [4,8,16,32,64]
time_for_update = np.zeros(max(batch_sizes) + 1)
train_cost_all_batch = {}
test_accuracy_all_batch = {}
train_size = len(trainX)

for batch_size in batch_sizes:
    # create model
    train, predict = model.create_3_layer_NN(decay, learning_rate, n_input, n_hidden, n_output, debug_parameter = True)
    
    # run model
    t = time.time()
    train_cost, test_accuracy = model.run_NN_model(train, predict, batch_size, trainX, trainY, testX, testY, epochs)

    time_for_update[batch_size] = 1000*(time.time()-t)/epochs
    train_cost_all_batch[batch_size] = train_cost
    test_accuracy_all_batch[batch_size] = test_accuracy

#Plots
dv.draw_multi_plot(range(epochs), train_cost_all_batch, 'iteration', 'cross-entropy', batch_sizes, "b=", 'training cost', figure_dir_path, 'pa2_cost.png')
dv.draw_multi_plot(range(epochs), test_accuracy_all_batch, 'iteration', 'cross-entropy', batch_sizes, "b=", 'test accuracy', figure_dir_path, 'pa2_accuracy.png')
dv.draw_plot(batch_sizes, time_for_update[batch_sizes], 'batch size', 'time for update in ms', 'time for a weight update', figure_dir_path, 'pa2_time_update.png')