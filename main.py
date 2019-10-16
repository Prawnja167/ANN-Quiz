import tensorflow as tf
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler #preprocessing
from sklearn.preprocessing import OneHotEncoder #ngubah target jd numeric
from sklearn.preprocessing import OrdinalEncoder #ubah string jd numeric
from sklearn.model_selection import train_test_split #split data training dan test

def load_dataset() :
    df = pd.read_csv("LifeExpectancy.csv")

    x_data = df[["Gender","Residential","Physical Activity (times per week)","Happiness"]]
    y_data = df[["Life Expectancy"]]

    return x_data, y_data

x_data, y_data = load_dataset()

#1. ubah string feature jd numeric feature
ord_enc = OrdinalEncoder()
x_data[["Gender", "Residential"]] = ord_enc.fit_transform(x_data[["Gender", "Residential"]])

#2. ubah target jd numeric
one_enc = OneHotEncoder(sparse=False)

y_data = one_enc.fit_transform(y_data)

#3. split train and test data
train_inp, test_inp, train_out, test_out = train_test_split(x_data, y_data, test_size=0.25)

#4. normalisasi input
scaler = MinMaxScaler()
scaler.fit(train_inp) #cari data yg plg banyak, lalu difit
train_inp = scaler.transform(train_inp)
test_inp = scaler.transform(test_inp)

#5. bentuk neural networknya
layers = {
    "num_input" : 4,
    "num_hidden" : 5,
    "num_output" : 3
}

#6. inisialisasi weight, bias, alpha, dan epoch
weight = {
    "hidden" : tf.Variable(tf.random_normal([layers["num_input"], layers["num_hidden"]])),
    "output" : tf.Variable(tf.random_normal([layers["num_hidden"], layers["num_output"]]))
}

bias = {
    "hidden" : tf.Variable(tf.random_normal([layers["num_hidden"]])),
    "output" : tf.Variable(tf.random_normal([layers["num_output"]]))
}

alpha = 0.1
epoch = 1000 #mau ningkatin accuracy : naikin epoch atau naikin jumlah hidden layer

#7. bikin placeholder buat input dan output
inp = tf.placeholder(tf.float32, [None, layers["num_input"]])
out_true = tf.placeholder(tf.float32, [None, layers["num_output"]])

#8. bikin activation function (pakai sigmoid) dan forward pass


#predict
def forward_pass() :
    #hitung dr input ke hidden
    wx1_b = tf.matmul(inp, weight["hidden"]) + bias["hidden"]
    #panggil activation function
    y1 = tf.nn.sigmoid(wx1_b)

    #hitung dr hidden ke output
    wx2_b = tf.matmul(y1, weight["output"]) + bias["output"]
    result = tf.nn.sigmoid(wx2_b)

    return result

out_prediction = forward_pass()

#9. hitung error lalu optimize (update weight dan bias pake gradient descent optimizer)
error = tf.reduce_mean(0.5 * (out_true - out_prediction) ** 2)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(error)

#kalau pake tf variable hrs diinisialisasi
init = tf.global_variables_initializer()
#kalau pake tf mau dapet hasil perhitungan hrs pake tf session
with tf.Session() as sess :
    sess.run(init)

    dict_train = {
        inp : train_inp,
        out_true: train_out
    }

    dict_test = {
        inp : test_inp,
        out_true : test_out
    }

    for i in range(epoch+1) : #pake +1 karna fornya di phyton itu < bukan <=
        sess.run(train, feed_dict = dict_train)

        #print accuracy
        if (i % 100 == 0) :
            matches = tf.equal(tf.argmax(out_prediction, axis = 1), tf.argmax(out_true, axis = 1))
            accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

            print("Iteration: {}\tAccuracy: {:.02f}%".format(i,sess.run(accuracy, feed_dict=dict_test) * 100))
