import numpy as np
import os
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import time
import data_preprocessor as dp 
import model 
import data_visualizer as dv

decay = 1e-6
learning_rate = 0.01
n_input = 36
n_hidden1 = 10
n_hidden2 = 10
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
np.random.seed(0)
train, predict = model.create_4_layer_NN(decay, learning_rate, n_input, n_hidden1, n_hidden2, n_output)


# train and test
t = time.time()
train_cost_4, test_accuracy_4 = model.run_NN_model(train, predict, batch_size, trainX, trainY, testX, testY, epochs)
print(test_accuracy_4[10])
print("4 Layer time: %f sec"%(time.time()-t))

#create train and predict function
train, predict = model.create_3_layer_NN(decay, learning_rate, n_input, n_hidden1, n_output)

# train and test
t = time.time()
train_cost_3, test_accuracy_3 = model.run_NN_model(train, predict, batch_size, trainX, trainY, testX, testY, epochs)
print(test_accuracy_3[10])
print("3 Layer time: %f sec"%(time.time()-t))

train_cost_all_layer = {3: train_cost_3, 4: train_cost_4}
test_accuracy_all_layer = {3: test_accuracy_3, 4: test_accuracy_4}

#Plots
dv.draw_multi_plot(range(epochs), train_cost_all_layer, 'iteration', 'cross-entropy', [3, 4], "layer=", 'training cost', figure_dir_path, 'pa5_cost.png')
dv.draw_multi_plot(range(epochs), test_accuracy_all_layer, 'iteration', 'accuracy', [3, 4], "layer=", 'test accuracy', figure_dir_path, 'pa5_accuracy.png')

