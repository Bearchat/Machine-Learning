import tensorflow as tf

"""
Tensorflow 自带 tensorboard ，可以自动显示我们所建造的神经网络流程图：
就是用 with tf.name_scope 定义各个框架，注意看代码注释中的区别：

"""


# add_layer 函数里面所有的with都是为了tensorboard添加上去的
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    # 区别:大框架，定义层layer，里面有小部件
    with tf.name_scope('layer'):
        # 区别：小部件
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs


# 这个就是在tensorboard上可视化的时候的区别：
# 使用with tf.name_scope('inputs')可以将xs和ys包含进来
# 形成一个大的图层，图层的名字就是with tf.name_scope()方法里的参数。
# 区别：大框架，里面有 inputs x，y
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')  # 这个name的属性，也是为了使用tensorboard，添加上来的
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')  # 同上

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# 同上，这里的with也是一样的
# 区别：定义框架 train
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

# 区别：定义框架 train
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

# 上面的wtih或者是name都是可选的，可以选择添加，也可以选择不添加，but下面的这一行是一定要写的。
# 这个表明了 在当前的目录下面创建一个名为logs的文件夹，然后把图的信息保存进去
# 这样运行完这段代码之后，就会有一个logs的文件夹被创建

# 区别：sess.graph 把所有框架加载到一个文件中放到文件夹"logs/"里
# 接着打开terminal，进入你存放的文件夹地址上一层，运行命令 tensorboard --logdir='logs/'
# 会返回一个地址，然后用浏览器打开这个地址，在 graph 标签栏下打开

if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
    writer = tf.train.SummaryWriter('logs/', sess.graph)
else:  # tensorflow version >= 0.12
    writer = tf.summary.FileWriter("logs/", sess.graph)

if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

sess.run(init)

# 这样运行完这段代码之后，就会有一个logs的文件夹被创建，然后在logs的文件夹的目录下面 执行tensorboard 就可以可视化了
# 执行完这段代码之后，在终端执行，下面两句：
#  cd D:\WorkSpace\GithubLocalRepository\Machine-Learning\tensorflow_tutorials
#  tensorboard --logdir=logs
