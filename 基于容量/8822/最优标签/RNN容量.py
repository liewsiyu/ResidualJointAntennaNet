""" Recurrent Neural Network.

"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from numpy import mat

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''
a = 50
dataset = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\8822\全信道矩阵(p=50行列).csv')).iloc[:, 1:]
dataset = np.asarray(dataset, np.float32)

label = pd.read_csv(open(r'D:\天线选择 郭志斌\正确数据集\8822\容量标签(p=50行列).csv')).iloc[:, 1]
label = np.asarray(label, np.int32)
label.astype(np.int32)
n_class = 784
n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))  # 样本，类别
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1  # 非零列赋值为1

xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.2, random_state=40)

# Training Parameters
learning_rate = 0.1
training_steps =  200000
#500000
batch_size = 100
display_step = 200

# Network Parameters
num_input = 8  # MNIST data input (img shape: 28*28)
timesteps = 8  # timesteps
num_hidden = 100  # hidden l ayer num of features
num_classes = n_class  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def main():
    logits = RNN(X, weights, biases)
    prediction = tf.nn.softmax(logits)
    print('prediction shape is', prediction)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    trainStart = time()
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(0, training_steps - 1):
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 原先
            # start = (step * batch_size) % (100000)
            # end = ((step * batch_size) + batch_size) % (100000)
            start = (step * batch_size) % 160000
            end = ((step * batch_size) + batch_size) % 160000

            if end == 0:
                end = start + 100

            batch_x = xTrain[start:end]
            batch_y = yTrain[start:end]
            # print("step is",step)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        print("Optimization Finished!")
        train = time() - trainStart

        # Calculate accuracy for 128 mnist test images
        testStart = time()
        test_len = 100
        test_data = xTest[:test_len].reshape((-1, timesteps, num_input))
        test_label = yTest[:test_len]
        print("Testing Accuracy:", \
              sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
        test = time() - testStart

        y_PreRes = prediction.eval(feed_dict={X: (xTest[0:40000].reshape((-1, timesteps, num_input)))})

        x = pd.read_csv(r'D:\天线选择 郭志斌\正确数据集\8822\8822快速查询数据表.csv', header=None)
        matrix = x.values

        def GetFrom(y_pre):
            for i in range(0, 784):
                if y_pre == matrix[i][0]:
                    return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i, 4]

        y_PreRes_np = np.array(y_PreRes)
        print('y_PreRes_np shape is', y_PreRes_np.shape)
        y_PreRes_np_Res = np.argmax(y_PreRes_np, axis=1) + 1

        fullChainCapacity = pd.read_csv(open(r"D:\天线选择 郭志斌\正确数据集\8822\全信道容量(p=50行列).csv")).iloc[:, 1:]
        fullChainCapacity = fullChainCapacity[0:200000]
        fullChainCapacity = np.asarray(fullChainCapacity, np.float32)

        subChainCapacity = pd.read_csv(open(r"D:\天线选择 郭志斌\正确数据集\8822\子信道容量(p=50行列).csv")).iloc[:, 1:]
        subChainCapacity = subChainCapacity[0:200000]
        subChainCapacity = np.asarray(subChainCapacity, np.float32)

        fullChainCapacity_Mean = np.mean(fullChainCapacity)
        subChainCapacity_Mean = np.mean(subChainCapacity)

        I = np.eye(8)
        I2 = np.eye(2)
        Loss = []
        Gain = []
        for i in range(40000):
            ArrayA = xTest[i].reshape(8, 8)
            ArrayA = np.matrix(ArrayA)
            i1, i2, j1, j2 = GetFrom(y_PreRes_np_Res[i])
            RNN_sub = mat(np.zeros((2, 2)), dtype=float)
            RNN_sub[0, 0] = ArrayA[i1, j1]
            RNN_sub[0, 1] = ArrayA[i1, j2]
            RNN_sub[1, 0] = ArrayA[i2, j1]
            RNN_sub[1, 1] = ArrayA[i2, j2]

            Full_Capacity = np.log2(np.linalg.det(I + a * ArrayA.T * ArrayA / 8))
            Sub_Capacity = np.log2(np.linalg.det(I2 + a * RNN_sub.T * RNN_sub / 2))

            Gain.append(Sub_Capacity)
            Loss.append(Full_Capacity - Sub_Capacity)

        Gain_Mean = np.mean(Gain)
        Loss_Mean = np.mean(Loss)
        Loss_Variance = np.var(Loss)

        if test < 1e-6:
            testUnit = "ns"
            test *= 1e9
        elif test < 1e-3:
            testUnit = "us"
            test *= 1e6
        elif test < 1:
            testUnit = "ms"
            test *= 1e3
        else:
            testUnit = "s"

        if train < 1e-6:
            trainUnit = "ns"
            train *= 1e9
        elif train < 1e-3:
            trainUnit = "us"
            train *= 1e6
        elif train < 1:
            trainUnit = "ms"
            train *= 1e3
        else:
            trainUnit = "s"

        print("基于信道容量,信噪比:",a)
        print("RNN(20000个测试样本)")
        print("80000个样本的训练时间 %.1f %s" % (train, trainUnit))
        print("20000个样本的测试时间 %.1f %s" % (test, testUnit))
        print("测试样本的全信道容量均值", fullChainCapacity_Mean)
        print("测试样本的穷举法子信道容量均值", subChainCapacity_Mean)
        print("穷举法损失均值", fullChainCapacity_Mean - subChainCapacity_Mean)
        print('预测准确率', accuracy)
        print('预测子信道容量均值', Gain_Mean)
        print('预测损失均值', Loss_Mean)
        print('预测损失方差', Loss_Variance)


if __name__ == "__main__":
    main()



