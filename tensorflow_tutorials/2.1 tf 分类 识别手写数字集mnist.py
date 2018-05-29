"""
搭建一个最简单的网络，只有输入层和输出层
输入数据的维度是 28*28 = 784 
输出数据的维度是 10个特征
激活函数使用softmax
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # 手写板 mnist 是 tensorflow自带的examples

"""

交叉熵 softmax回归 梯度下降
https://www.jianshu.com/p/4195577585e6
https://zhuanlan.zhihu.com/p/28767527


文件说明:
https://www.cnblogs.com/imlvbu/p/7738543.html

数据中包含55000张训练图片，每张图片的分辨率是28×28，
所以我们的训练网络输入应该是28×28=784个像素数据。

搭建一个最简单的训练网络结构，只有输入层和输出层。

输入 每个特征都是一个节点(神经元)

loss函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
其公式:https://blog.csdn.net/lanchunhui/article/details/61413557

"""
# number 1 to 10 data
# one_hot 表示的是一个长度为n的数组，只有一个元素是1.0，其他元素都是0.0 
# 比如在n=4的情况下，标记2对用的one_hot 的标记是 [0.0 , 0.0 , 1.0 ,0.0]
# 使用 one_hot 的直接原因是，我们使用 0～9 个类别的多分类的输出层是 softmax 层
# softmax 它的输 出是一个概率分布，从而要求输入的标记也以概率分布的形式出现，进而可以计算交叉熵

# 用代码再把它解析出来然后转成向量数组便于tensorflow引用, 返回一个 Datasets对象
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



def add_layer(inputs, in_size, out_size, activation_function=None, ):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs


# (mnist.test.images, mnist.test.labels)
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    # tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
    # tf.argmax (y_pre,1) 返回每一行 下标最大的元素，1表示按行
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# define placeholder for inputs to network
# 每张图片的分辨率是28×28，所以我们的训练网络输入应该是28×28=784个像素数据。None表示可以输入任意多个样本
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])  # 每张图片都表示一个数字，所以我们的输出是数字0到9，共10类。

# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# loss函数（即最优化目标函数）选用交叉熵函数
# 交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
# prediction 表示预测值 ，ys表示真实的输出

#### 感觉这样写 ，也应该可以 ： cross_entroy = tf.reduce_mean( tf.nn.softmax_cross_entroy_with_logits(ys,prediction))
#### 用tf.nn.softmax_cross_entropy_with_logits 来计算预测值y与真实值y_的差值，并取均值 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

sess.run(init)

# 当调整循环次数时可以发现总训练的样本数越多，精度就越高
for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 每次只取100张图片，免得数据太多训练太慢。#在每次循环中我们都随机抓取训练数据中 100 个数据点
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 100 == 0:
        # 注意，这里改成了测试集
        # print('loss -> cross_entropy : ',sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys}))
        print("第{}次，准确率为{}".format(i + 100, compute_accuracy(mnist.test.images, mnist.test.labels)))

"""
输出：
0.0841
0.6386
0.7359
0.7792
0.7999
0.8145
0.8395
0.8491
0.8509
0.8584
0.8645
0.8653
0.865
0.868
0.8671
0.8738
0.8716
0.8793
0.8312
0.879
"""
