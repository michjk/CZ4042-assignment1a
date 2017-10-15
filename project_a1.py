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
batch_size = 32

dataset_dir_path = "dataset"
train_data_path = os.path.join(dataset_dir_path, "sat_train.txt")
test_data_path = os.path.join(dataset_dir_path, "sat_test.txt")
figure_dir_path = "figure"

#read train data
trainX, trainY, X_min, X_max = dp.load_data(train_data_path)

#read test data
testX, testY, _, _ = dp.load_data(test_data_path, True, X_min, X_max)

print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)

#create train and predict function
train, predict = model.create_3_layer_NN(decay, learning_rate, n_input, n_hidden, n_output)

# train and test
train_cost, test_accuracy = model.run_NN_model(train, predict, batch_size, trainX, trainY, testX, testY, epochs)

#Plots
dv.draw_plot(range(epochs), train_cost, 'iteration', 'cross-entropy', 'training cost', figure_dir_path, 'pa1_cost.png')
dv.draw_plot(range(epochs), test_accuracy, 'iteration', 'accuracy', 'test accuracy', figure_dir_path, 'pa1_accuracy.png')
