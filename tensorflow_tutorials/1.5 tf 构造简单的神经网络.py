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
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""


https://www.jianshu.com/p/e112012a4b2d (整体)
https://blog.csdn.net/ma524654165/article/details/77600362 (细节)

神经网络解决问题步骤： 
1.提取问题中实体的特征向量作为神经网络的输入。也就是说要对数据集进行特征工程，然后知道每个样本的特征维度，以此来定义输入神经元的个数。 
2.定义神经网络的结构，并定义如何从神经网络的输入得到输出。也就是说定义输入层，隐藏层以及输出层。 
3.通过训练数据来调整神经网络中的参数取值，这是训练神经网络的过程。一般来说要定义模型的损失函数，以及参数优化的方法，如交叉熵损失函数和梯度下降法调优等。 
4.利用训练好的模型预测未知的数据。也就是评估模型的好坏。


搭建神经网络的基本流程

定义添加神经层的函数

1.训练的数据
2.定义节点准备接收数据
3.定义神经层：隐藏层和预测层
4.定义 loss 表达式
5.选择 optimizer 使 loss 达到最小

"""
# 添加层
# 神经网络的基本构造是要有输入，还要有输入映射到下一层的权重和偏差，
# 最后神经元还有一个激活函数（这个有没有看需求），控制输出
def add_layer(inputs, in_size, out_size, activation_function=None):
    """
    :param inputs:  输入unit
    :param in_size: 映射到下一层的权重
    :param out_size: 映射到下一层的偏差
    :param activation_function: 激活函数
    :return:
    """
    # add one more layer and return the output of this layer
    # 定义出参数 Weights，biases，拟合公式 Wx_plus_b
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # tf.random_normal  从服从指定正太分布的数值中取出指定个数的值
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 构造一个数据集(训练的数据,为了更加符合实际，我们会加一些噪声)
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 创建从-1到1的等差数列 个数为300  然后在转化成300 x 1的矩阵
noise = np.random.normal(0, 0.05, x_data.shape)  # 高斯分布的概率密度函数的numpy表示  三个参数分别是均值 标准差 shape
y_data = np.square(x_data) - 0.5 + noise  # 加上噪音 倒U曲线更离散

# 画出上面原始数据的图形,可视化输入数据集
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(x_data, y_data)
# print('x_data :\n', x_data)
# print('y_data :\n', y_data)
# plt.show()


# placeholder 占个位(定义节点准备接收数据 -- 输入层)
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 定义神经层:隐藏层和预测层
# add hidden layer  输入值是xs, 在隐藏层有10个神经元,激活函数使用 tf.nn.relu
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
# 上一层的输出是这一层的输入  输入值是隐藏层 l1,在预测层输出1个结果
prediction = add_layer(l1, 10, 1, activation_function=None)

# 定义loss函数
# the error between prediction and real data
# loss函数和使用梯度下降的方式来求解
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

optimizer = tf.train.GradientDescentOptimizer(0.1)  # 选择 Gradient Descent 这个最基本的 Optimizer：
train_step = optimizer.minimize(loss)  # 神经网络的 key idea，就是让 loss 达到最小：

# important step  对所有变量进行初始化
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12

if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

sess = tf.Session()

# 上面的定义的并没有开始运算,知道sess.run 蔡开始运算
sess.run(init)


# 画图查看拟合效果  有问题!!!
fig = plt.figure()
bx = fig.add_subplot(1, 1, 1)
bx.scatter(x_data, y_data)  # scatter 散点图
plt.show()

# 迭代1000次学习，sess.run.optimizer
for i in range(1000):
    # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

    if i % 50 == 0:
        # to see the step improvement
        # 在带有placeholder的变量里面，每一次sess.run 都需要给一个feed_dict，这个不能省略啊！
        print("loss : ", sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

        bx.plot(x_data, prediction, 'g-', lw=5)  # plot线图 # AttributeError: 'Tensor' object has no attribute 'ndim'
        plt.xlabel('x_data')
        plt.ylabel('y_data')
        plt.show()

# # 画图查看拟合效果  有问题!!!
# fig = plt.figure()
# bx = fig.add_subplot(1, 1, 1)
# bx.scatter(x_data, y_data)  # scatter 散点图
# bx.plot(x_data, prediction, 'g-', lw=6)  # plot线图
# plt.xlabel('x_data')
# plt.ylabel('y_data')
# plt.show()
