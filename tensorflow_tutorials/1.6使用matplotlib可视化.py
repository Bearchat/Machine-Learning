"""
构造一个3层的网络
输入层一个结点，隐层3个结点，输出层一个结点

输入层的维度是[n,1]
隐层的维度是  [1,10]
输出层的维度是[10,1]

so,
权值矩阵的维度是：
weight1=[1,10]
bais1=[10,1]

weight2=[10,1]
bais2=[1,1]

网络的结构和1.5的内容是一样的，只不过是这一次把每次的训练结构可视化show出来了
https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/3-3-visualize-result/

https://github.com/MorvanZhou/Tensorflow-Tutorial/blob/master/tutorial-contents/301_simple_regression.py

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# important step
sess = tf.Session()
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# plot the real data
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data, y_data)
# plt.ion()#本次运行请注释，全局运行不要注释
# plt.show()


for i in range(1000):
    # training
    # sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

    _, l, pred = sess.run([train_step, loss, prediction], {xs: x_data, ys: y_data})  # 方法二
    if i % 50 == 0:
        # 密集线簇
        # prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # # plot the prediction
        # plt.plot(x_data, prediction_value)

        # 方法一: 动态展示拟合曲线
        # to visualize the result and improvement
        # try:
        #     ax.lines.remove(lines[0])
        # except Exception:
        #     pass
        # prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # # plot the prediction
        # lines = ax.plot(x_data, prediction_value, 'r-', lw=3)
        # plt.pause(0.1)

        # 方法二: 动态展示拟合曲线 散点图和线图放到一块
        # https://github.com/MorvanZhou/Tensorflow-Tutorial/blob/master/tutorial-contents/301_simple_regression.py
        plt.cla()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()  # 图形展示完之后不会自动退出关掉
plt.show()
