import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

def loadDataset() :
    dataset = pd.read_csv("monthly-milk-production.csv", index_col="Month")

    return dataset

dataset = loadDataset()
dataset.index = pd.to_datetime(dataset.index)

time_step = 12
test_size = 12

train_dataset = dataset.head(len(dataset) - test_size)
test_dataset = dataset.tail(test_size)

alpha = 0.1
epoch = 10000
batch_size = 2

num_input = 1
num_output = 1
num_hidden = 100

scaler = MinMaxScaler()
scaler.fit(train_dataset)
normalized_train_dataset = scaler.transform(train_dataset)
__ = scaler.transform(test_dataset)

cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden, activation=tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_output, activation=tf.nn.sigmoid)

inp = tf.placeholder(tf.float32, [None, time_step, num_input])
target_output = tf.placeholder(tf.float32, [None, time_step, num_output])

output, state = tf.nn.dynamic_rnn(cell, inp, dtype=tf.float32)

error = tf.reduce_mean(0.5 * (target_output - output) ** 2)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def nextBatch(dataset, batch_size, time_step) :
    x_batch = np.zeros(shape=(batch_size, time_step, num_input))
    y_batch = np.zeros(shape=(batch_size, time_step, num_output))

    for i in range(batch_size) :
        start = np.random.randint(0, len(dataset) - time_step)
        x_batch[i] = dataset[start:start+time_step]
        y_batch[i] = dataset[start+1:start+time_step+1]

    return x_batch, y_batch

with tf.Session() as sess :
    sess.run(init)

    for i in range(epoch+1) :
        inp_batch, tar_batch = nextBatch(normalized_train_dataset, batch_size, time_step)
        train_dict = {
            inp : inp_batch,
            target_output : tar_batch
        }
        sess.run(train, feed_dict=train_dict)

        if (i%2000 == 0) :
            print_error = sess.run(error, feed_dict=train_dict)
            print("Iteration: {}, Error: {}".format(i, print_error))

    saver.save(sess, "model/rnn-model.ckpt")

with tf.Session() as sess:
    saver.restore(sess, "model/rnn-model.ckpt")
    seed = list(normalized_train_dataset)

    for i in range(test_size) :
        inp_batch = np.array(seed[-time_step:]).reshape(1, time_step, num_input)
        test_dict = {
            inp : inp_batch
        }
        
        prediction = sess.run(output, feed_dict=test_dict)
        seed.append(prediction[0,-1,0])

result = scaler.inverse_transform(np.array(seed[-time_step:]).reshape(time_step,1))
test_dataset['Predict'] = result
test_dataset.plot()
plt.show()