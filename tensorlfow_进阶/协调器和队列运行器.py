import tensorflow as tf

images = ['img1', 'img2', 'img3', 'img4', 'img5']
labels = [1, 2, 3, 4, 5]

# 迭代次数
epoch_num = 10
# tf.train.slice_input_producer是一个tensor生成器，
# 作用是按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。
f = tf.train.slice_input_producer([images, labels], num_epochs=2, shuffle=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 这个必须有局部变量f属于
    sess.run(tf.local_variables_initializer())
    # 开启一个协调器
    coord = tf.train.Coordinator()
    # 还需要调用tf.train.start_queue_runners 函数来启动执行文件名队列填充的线程
    # 使用start_queue_runners 启动队列填充
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(epoch_num):
        k = sess.run(f)
        print('************************')
        print(i, k)
    # 协调器coord发出所有线程终止信号
    coord.request_stop()
    # 把开启的线程加入主线程，等待threads结束
    coord.join(threads)
    print("Done Well !!!")
