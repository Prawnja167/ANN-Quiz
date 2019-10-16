import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

def load_dataset() :
    dataset = pd.read_csv("LifeExpectancy.csv")
    x_data = dataset[["Gender", "Residential", "Physical Activity (times per week)", "Happiness"]] 
    y_data = dataset[["Life Expectancy"]]

    return x_data, y_data

x_data, y_data = load_dataset()

ord_enc = OrdinalEncoder()
x_data[["Gender", "Residential"]] = ord_enc.fit_transform(x_data[["Gender", "Residential"]])

one_enc = OneHotEncoder(sparse = False)
y_data = one_enc.fit_transform(y_data)

train_inp, test_inp, train_out, test_out = train_test_split(x_data, y_data, test_size = 0.25)

scaler = MinMaxScaler()
scaler.fit(train_inp)
train_inp = scaler.transform(train_inp)
test_inp = scaler.transform(test_inp)

alpha = 0.5
epoch = 10000

layers = {
    "num_input" : 4,
    "num_hidden1" : 5,
    "num_hidden2" : 5,
    "num_output" : 3
}

weights = {
    "hidden1" : tf.Variable(tf.random_normal([layers["num_input"], layers["num_hidden2"]])),
    "hidden2" : tf.Variable(tf.random_normal([layers["num_hidden1"], layers["num_hidden2"]])),
    "output" : tf.Variable(tf.random_normal([layers["num_hidden2"], layers["num_output"]]))
}

biases = {
    "hidden1" : tf.Variable(tf.random_normal([layers["num_hidden1"]])),
    "hidden2" : tf.Variable(tf.random_normal([layers["num_hidden2"]])),
    "output" : tf.Variable(tf.random_normal([layers["num_output"]]))
}

inp = tf.placeholder(tf.float32, [None, layers["num_input"]])
out_true = tf.placeholder(tf.float32, [None, layers["num_output"]])

def forward_pass() :
    wx1b = tf.matmul(inp, weights["hidden1"]) + biases["hidden1"]
    y1 = tf.nn.sigmoid(wx1b)
    wx2b = tf.matmul(y1, weights["hidden2"]) + biases["hidden2"]
    y2 = tf.nn.sigmoid(wx2b)
    wx3b = tf.matmul(y2, weights["output"]) + biases["output"]
    result = tf.nn.sigmoid(wx3b)

    return result

out_prediction = forward_pass()

error = tf.reduce_mean(0.5 * (out_true - out_prediction) ** 2)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess : 
    sess.run(init)

    train_dict = {
        inp : train_inp,
        out_true : train_out
    }

    test_dict = {
        inp : test_inp,
        out_true : test_out
    }

    for i in range(epoch + 1) :
        sess.run(train, feed_dict = train_dict)
        if i % 1000 == 0 :
            matches = tf.equal(tf.argmax(out_prediction, axis = 1), tf.argmax(out_true, axis = 1))
            accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
            print("Iteration : {}\tAccuracy {:0.2f}%".format(i, sess.run(accuracy, feed_dict = test_dict) * 100))
