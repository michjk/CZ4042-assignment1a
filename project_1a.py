import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

def init_weights(n_in=1, n_out=1, logistic=True, weight_num=1):
    W_values = np.asarray(
        np.random.uniform(
        low=-np.sqrt(6. / (n_in + n_out)),
        high=np.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)),
        dtype=np.float32
        )
    if logistic == True:
        W_values *= 4
    print(weight_num)
    return tf.Variable(W_values, dtype=tf.float32, name="W"+str(weight_num))

def init_bias(n=1, bias_num=1):
    return tf.Variable(np.zeros(n), dtype=tf.float32, name="b"+str(bias_num))

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-np.min(X, axis=0))

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels



#parameter
decay = 1e-6
learning_rate = 0.01
epochs = 1000
batch_size = 32

input_layer_size = 36
hidden_layer_size = 10
output_layer_size = 6

X = tf.placeholder(tf.float32, shape=[None, input_layer_size], name="X")
Y = tf.placeholder(tf.float32, shape=[None, output_layer_size], name="Y")

w1, b1 = init_weights(input_layer_size, hidden_layer_size, weight_num=1), init_bias(hidden_layer_size, bias_num=1)
w2, b2 = init_weights(hidden_layer_size, output_layer_size, logistic=False, weight_num=2), init_bias(output_layer_size, bias_num=2)

h1 = tf.nn.sigmoid(tf.matmul(X, w1) + b1, name="sigmoid")
y2 = tf.matmul(h1, w2) + b2
py = tf.nn.softmax(y2, name="softmax")

y_x = tf.argmax(py, axis=1, name="max")

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y2)) \
    + decay*tf.reduce_sum(tf.square(w1) + tf.reduce_sum(tf.square(w2)))

updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#read train data
train_input = np.loadtxt('sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainX_min, trainX_max = np.min(trainX, axis=0), np.max(trainX, axis=0)
trainX = scale(trainX, trainX_min, trainX_max)

train_Y[train_Y == 7] = 6
trainY = np.zeros((train_Y.shape[0], 6))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1


#read test data
test_input = np.loadtxt('sat_test.txt',delimiter=' ')
testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)

testX_min, testX_max = np.min(testX, axis=0), np.max(testX, axis=0)
testX = scale(testX, testX_min, testX_max)

test_Y[test_Y == 7] = 6
testY = np.zeros((test_Y.shape[0], 6))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1

print(trainX.shape, trainY[0])
print(testX.shape, testY.shape)

# train and test
n = len(trainX)
test_accuracy = []
train_cost = []

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(epochs):
    #print(i)
    
    trainX, trainY = shuffle_data(trainX, trainY)
    sum_cost = 0.0
    for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
        feed_dict = {X:trainX[start:end], Y:trainY[start:end]}
        tmp_cost, _ = sess.run([cost, updates], feed_dict=feed_dict)
        sum_cost += tmp_cost
    last_cost = sum_cost/(n // batch_size)
    train_cost = np.append(train_cost, last_cost)
    predict = sess.run(y_x, feed_dict={X:testX})
    last_acc = np.mean(np.argmax(testY, axis=1) == predict)
    test_accuracy = np.append(test_accuracy, last_acc)
    #print('train cost: %.1f test accuracy: %.1f'%(last_cost, last_acc))

print('%.1f accuracy at %d iterations'%(np.max(test_accuracy)*100, np.argmax(test_accuracy)+1))

#Plots
plt.figure()
plt.plot(range(epochs), train_cost)
plt.xlabel('iterations')
plt.ylabel('cross-entropy')
plt.title('training cost')
plt.savefig('p1a_sample_cost.png')

plt.figure()
plt.plot(range(epochs), test_accuracy)
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.title('test accuracy')
plt.savefig('p1a_sample_accuracy.png')

plt.show()


