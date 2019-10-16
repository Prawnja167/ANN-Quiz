import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler # normalisasi
from sklearn.preprocessing import OneHotEncoder # ubah target jadi numeric
from sklearn.preprocessing import OrdinalEncoder # ubah string jadi numeric
from sklearn.model_selection import train_test_split

def load_dataset():
    df = pd.read_csv("LifeExpectancy.csv")

    x_data = df[["Gender","Residential","Physical Activity (times per week)","Happiness"]]
    y_data = df[["Life Expectancy"]]

    return x_data, y_data


# ubah string feature jadi numeric feature

x_data, y_data = load_dataset()

ord_enc = OrdinalEncoder()
# ord_enc.fit(x_data[["Gender", "Residential"]])
x_data[["Gender", "Residential"]] = ord_enc.fit_transform(x_data[["Gender", "Residential"]])


# ubah target jadi numeric
one_enc = OneHotEncoder(sparse=False)

# one_enc.fit(y_data)

y_data = one_enc.fit_transform(y_data)


# split data jadi train & test
train_input, test_input, train_out, test_out = train_test_split(x_data, y_data, test_size=0.25)


# normalisasi input
scaler = MinMaxScaler()
scaler.fit(train_input)
train_input = scaler.transform(train_input)
test_input = scaler.transform(test_input)


# bentuk neural network (layers)
layers = {
    "num_input": 4,
    "num_hidden": [5, 6],
    "num_output": 3
}


# inisialisasi weight, bias, alpha, epoch

weight = {
    'hidden': tf.Variable(tf.random_normal([layers["num_input"], layers["num_hidden"][0]])), # atau 4 (layer input), 5 (layer hidden)
    'hidden2': tf.Variable(tf.random_normal([layers["num_hidden"][0], layers["num_hidden"][1]])),
    'output': tf.Variable(tf.random_normal([layers["num_hidden"][1], layers["num_output"]])) # atau 5 (layer hidden), 3 (layer output)
}

bias = {
    'hidden': tf.Variable(tf.random_normal([layers["num_hidden"][0]])),
    'hidden2': tf.Variable(tf.random_normal([layers["num_hidden"][1]])),
    'output': tf.Variable(tf.random_normal([layers["num_output"]]))
}

alpha = 0.5
epoch = 10000


# thresholder input & output

inp = tf.placeholder(tf.float32, [None, layers["num_input"]])
output_true = tf.placeholder(tf.float32, [None, layers["num_output"]])

# activation function & forward pass (sigmoid activation function)

# predict
def forward_pass():
    # input -> hidden
    Wx1_b = tf.matmul(inp, weight["hidden"]) + bias["hidden"]
    # panggil act func.
    y1 = tf.nn.sigmoid(Wx1_b)

    # hidden1 -> hidden2
    Wx2_b = tf.matmul(y1, weight["hidden2"]) + bias["hidden2"]
    # panggil act func.
    y2 = tf.nn.sigmoid(Wx2_b)

    # hidden -> output
    Wx3_b = tf.matmul(y2, weight["output"]) + bias["output"]
    result = tf.nn.sigmoid(Wx3_b)

    return result

prediction_out = forward_pass()

# hitung error & optimize (weight & bias update)
# GradicentDescentOptimizer
# error = 0.5 * (target - result(y)) ^ 2
error = tf.reduce_mean(0.5 * (output_true - prediction_out)**2)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(error)


# krna pake tf.Variable, maka perlu di inisialisasi

init = tf.global_variables_initializer()

# krna pake tf, maka utk hasil perhitungan pakai tf.Session()

# training
# sess = tf.Session()
with tf.Session() as sess:
    # dalam scope
    sess.run(init)

    dict_train = {
        inp: train_input,
        output_true: train_out
    }

    dict_test = {
        inp: test_input,
        output_true: test_out
    }

    for i in range(epoch+1):
        sess.run(train, feed_dict=dict_train)

        if i % 1000 == 0:
            matches = tf.equal(tf.argmax(prediction_out, axis=1), tf.argmax(output_true, axis=1))

            accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

            print("Itteration: {}\tAccuracy: {}%".format(i, sess.run(accuracy, feed_dict=dict_test ) * 100))