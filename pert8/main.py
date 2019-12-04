import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_dataset() :
    #sep="|"  //separator
    #names=["",""] //ganti nama th nya
    #index_col=["",""] //jadiin data jd index (sb x)
    #usecols=["",""] //col apa aja yg dipake
    #dtype={"column name" : "float64/int64/Int64"}
    
    dataset = pd.read_csv("Carbon Dioxide Emissions in Hawaii.csv", parse_dates={"timestamp" : ["Year", "Month"]}, index_col="timestamp")

    return dataset

dataset = load_dataset()

time_step = 3
test_size = int(384*30/100)
train_dataset = dataset.head(len(dataset)-test_size)
test_dataset = dataset.tail(test_size)

alpha = 0.1
epoch = 1000
batch_size = 4

num_input = 1
num_hidden = 5
num_output = 1

scaler = MinMaxScaler()
scaler.fit(train_dataset)
normalized_training_dataset = scaler.transform(train_dataset)
__ = scaler.transform(test_dataset)

inp = tf.placeholder(tf.float32, [None, time_step, num_input])
target_output = tf.placeholder(tf.float32, [None, time_step, num_output])

cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden, activation=tf.nn.sigmoid)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_output, activation=tf.nn.sigmoid)

output, state = tf.nn.dynamic_rnn(cell, inp, dtype=tf.float32)

error = tf.reduce_mean(0.5 * (target_output - output) ** 2)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def next_batch(dataset, batch_size, time_step) :
    x_batch = np.zeros(shape=(batch_size, time_step, num_input))
    y_batch = np.zeros(shape=(batch_size, time_step, num_output))

    for i in range(batch_size) :
        start = np.random.randint(0, len(dataset)-time_step)
        x_batch[i] = dataset[start:start+time_step]
        y_batch[i] = dataset[start+1:start+time_step+1]

    return x_batch, y_batch

with tf.Session() as sess :
    sess.run(init)

    for i in range(epoch+1) :
        inp_batch, tar_batch = next_batch(normalized_training_dataset, batch_size, time_step)
        train_dict = {
            inp : inp_batch,
            target_output : tar_batch
        }

        sess.run(train, feed_dict=train_dict)

        if (i%200 == 0) :
            saver.save(sess, "model/rnn-model.ckpt")
        
        if (i%250 == 0) :
            print_error = sess.run(error, feed_dict=train_dict)
            print("Iteration: {}, Error: {}".format(i, print_error))

with tf.Session() as sess:
    saver.restore(sess, "model/rnn-model.ckpt")
    seed_data = list(normalized_training_dataset)

    for i in range(test_size) :
        inp_batch = np.array(seed_data[-time_step:]).reshape(1, time_step, num_input)
        test_dict = {
            inp : inp_batch
        }

        prediction = sess.run(output, feed_dict=test_dict)
        seed_data.append(prediction[0,-1,0])

result = scaler.inverse_transform(np.array(seed_data[-time_step:]).reshape(time_step,1))
print(len(result))
print(len(test_dataset))
trim = test_dataset.tail(time_step)
trim['Predict'] = result
trim.plot()
plt.show()


