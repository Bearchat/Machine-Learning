import tensorflow as tf

"""
placeholder 是 Tensorflow 中的占位符
要给节点输入数据时用 placeholder，在 TensorFlow 中用placeholder 来描述等待输入的节点，只需要指定类型即可，
然后在执行节点的时候用一个字典来“喂”这些节点。相当于先把变量 hold 住，然后每次从外部传入data，以这种形式传输数据 sess.run(***, feed_dict={input: **}).

注意 placeholder 和 feed_dict 是绑定用的。

这里简单提一下 feed 机制， 给 feed 提供数据，作为 run()
调用的参数， feed 只在调用它的方法内有效, 方法结束, feed 就会消失。

"""

# 定义两个等待输入的节点
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

# 需要传入的值放在了feed_dict={} 并一一对应每一个 input.
# placeholder 与 feed_dict={} 是绑定在一起出现的。

# 这里没有变量，就不需要 init =tf.global_variables_initializer() 这一步了
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
