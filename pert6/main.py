import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_dataset() :
    dataset = pd.read_csv("monthly-milk-production.csv", index_col = "Month", parse_dates=["Month"])
    return dataset

dataset = load_dataset()

# dataset.index = pd.to_datetime(dataset.index) #kalo gapake parse_dates

# dataset.plot()
# plt.show()

#inisialisasi time step (testing size)
#variabelnya dibedain supaya gabingung
#mau prediksi 21 bulan kedepan : time stepnya 12
time_step = 12
testing_size = 12
#ambil data dari depan sejumlah dataset-testing size
training_dataset = dataset.head(len(dataset) - testing_size)
#ambil 12 data dr blakang
test_dataset = dataset.tail(testing_size)

#normalisasi
scaler = MinMaxScaler()
scaler.fit(training_dataset)
normalized_training_dataset = scaler.transform(training_dataset)
#test dataset gaboleh ditimpa krn ntr mo dipake lg
__ = scaler.transform(test_dataset)

#bikin neural network
num_input = 1
num_output = 1
# num_hidden = 20
num_hidden = 100
#coba hidden jadi 100 trus jalanin, bakal kena gradient vanishing problem

#bikin alpha, epoch, batch_size
alpha = 0.1
epoch = 10000
batch_size = 2 #angkanya seminimal mungkin

#cell RNN
#kekurangan sigmoid : hasil trainingnya bisa jadi gradient vanishing : ketika angkanya kecil bs jadi 0 krn pembulatan
# cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden, activation = tf.nn.sigmoid)
#if hidden layernya 100, sigmoidnya ganti jd relu aja
cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden, activation = tf.nn.relu) #create rnn
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_output, activation = tf.nn.sigmoid) #bungkus rnn supaya outputnya sizenya = num_output

#bikin placeholder
inp = tf.placeholder(tf.float32, [None, time_step, num_input])
target_output = tf.placeholder(tf.float32, [None, time_step, num_output])

#bikin RNN jadi dynamic RNN buat ngilangin out of memory error krn graph/output data kegedean
output, state = tf.nn.dynamic_rnn(cell, inp, dtype = tf.float32)

#bikin error function & optimisasi
error = tf.reduce_mean(0.5 * (target_output-output) ** 2)
# optimizer = tf.train.AdamOptimizer(alpha)
#kalo hidden jd 100, ganti adam jadi gradient descent
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def next_batch(dataset, time_step, batch_size) :
    x_batch = np.zeros(shape=(batch_size, time_step, num_input))
    y_batch = np.zeros(shape=(batch_size, time_step, num_output))

    for i in range(batch_size) :
        start = np.random.randint(0, len(dataset)-time_step)
        x_batch[i] = dataset[start : start+time_step]
        y_batch[i] = dataset[start+1 : start+time_step+1] #+1 karna mau predict bulan selanjutnya
    
    return x_batch, y_batch

#cuma dijalanin 1x, abis training dikomen aja supaya bs jalanin sess yg bawah
with tf.Session() as sess :
    sess.run(init)

    for i in range(epoch+1) :
        inp_batch, tar_batch = next_batch(normalized_training_dataset, time_step, batch_size)
        train_dict = {
            inp : inp_batch,
            target_output : tar_batch
        }
        sess.run(train, feed_dict=train_dict)

        if i%200 == 0 :
            #BELAJAR CARA PRINT ACCURACY JUGA
            print_error = sess.run(error, feed_dict=train_dict)
            print("Iteration: {}, Error: {}".format(i, print_error))

    #save
    saver.save(sess, "rnn-model/my_model.ckpt")

with tf.Session() as sess:
    saver.restore(sess, "rnn-model/my_model.ckpt")
    seed_data = list(normalized_training_dataset)
    #input batch
    #shape = [None, time_step, num_input]
    #None->batch size
    for i in range(testing_size) :
        inp_batch = np.array(seed_data[-time_step:]).reshape(1, time_step, num_input)
        test_dict = {
            inp : inp_batch
        }
        prediction = sess.run(output, feed_dict=test_dict)
        seed_data.append(prediction[0,-1,0])

#reverse data yg udh dinormalize dr 0..1 jadi data awal
#output
#shape = [12,12,1]
#shape = [[output],[output]..sampe 12x]
result = scaler.inverse_transform(np.array(seed_data[-time_step:]).reshape([time_step,1]))

test_dataset['Prediction'] = result
test_dataset.plot()
plt.show()
