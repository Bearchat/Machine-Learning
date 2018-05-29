"""
训练数据的读取
https://blog.csdn.net/lin453701006/article/details/79402976
"""

import tensorflow as tf
import numpy as np
import os  # os模块包含操作系统相关的功能，可以处理文件和目录这些我们日常手动需要做的操作。因为我们需要获取test目录下的文件，所以要导入os模块。
import matplotlib.pyplot as plt

# 获取文件路径和标签
"""
数据集由训练数据和测试数据组成，训练数据包含猫和狗各12500张图片，测试数据包含12500张猫和狗的图片

主要流程： 
1.读取数据集，根据文件名，分成cat和dog两类图片和标签。这里cat和dog各有12500幅图片。 
2.使用np.hstack()将cat和dog的图片和标签整合为列表image_list和label_list，image_list和label_list的大小均为25000。 
3.将image_list和label_list合并，存放在temp中，此时temp的大小为2x25000。对temp进行转置，temp的大小变为25000x2。 
4.使用np.random.shuffle()打乱图片和标签。 
5.从temp中取出乱序后的image_list和label_list列向量并返回。

"""


# 读取数据和标签
def get_files(file_dir):
	cats = []
	label_cats = []
	dogs = []
	label_dogs = []
	for file in os.listdir(file_dir):  # 返回文件名
		name = file.split(sep = '.')  # 文件名按.分割
		if name[0] == 'cat':  # 如果是cat，标签为0，dog为1
			cats.append(file_dir + file)
			label_cats.append(0)
		else:
			dogs.append(file_dir + file)
			label_dogs.append(1)
	print('There are %d cats\nThere are %d dogs' % (len(cats), len(dogs)))  # 打印猫和狗的数量

	image_list = np.hstack((cats, dogs))
	label_list = np.hstack((label_cats, label_dogs))

	temp = np.array([image_list, label_list])
	temp = temp.transpose()
	np.random.shuffle(temp)  # 打乱图片

	image_list = list(temp[:, 0])
	label_list = list(temp[:, 1])
	label_list = [int(i) for i in label_list]  # 将label_list中的数据类型转为int型

	return image_list, label_list


# get_files(r'D:\WorkSpace\PythonHome\Machine-Learning\dogs-vs-cats\data\train')


"""
由于数据集较大，需要分批次通过网络。get_batch()就是用于将图片划分批次。

主要流程： 
1.image和label为list类型，转换为TensorFlow可以识别的tensor格式。 
2.使用tf.train.slice_input_producer()将image和label合并生成一个队列，然后从队列中分别取出image和label。其中image需要使用tf.image.decode_jpeg()进行解码，由于图片大小不统一，使用tf.image.resize_image_with_crop_or_pad()进行裁剪/扩充，最后使用tf.image.per_image_standardization()进行标准化，此时的image的shape为[208 208 3]。 
3.因为之前已经进行了乱序，使用tf.train.batch()生成批次，最后得到的image_batch和label_batch的shape分别为[1 208 208 3]和[1]。 
4.这里原作者代码中对label_batch又进行reshape，是多余的，删除后无影响。最终返回image_batch和label_batch
"""


# 将图片分批次
def get_batch(image, label, image_W, image_H, batch_size, capacity):
	'''''
	Args:
		image: list type
		label: list type
		image_W: image width
		image_H: image height
		batch_size: batch size
		capacity: the maximum elements in queue
	Returns:
		image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
		label_batch: 1D tensor [batch_size], dtype=tf.int32
	'''
	# image和label为list类型，需要进行数据类型转换
	image = tf.cast(image, tf.string)
	label = tf.cast(label, tf.int32)

	# make an input queue 把image和label合并生成一个队列
	input_queue = tf.train.slice_input_producer([image, label])

	label = input_queue[1]  # 读取label
	image_contents = tf.read_file(input_queue[0])  # 读取图片
	image = tf.image.decode_jpeg(image_contents, channels = 3)  # 解码图片

	######################################
	# data argumentation should go to here
	######################################

	# 因为图片大小不一致，需要进行裁剪/扩充
	image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

	# 按照原代码使用标准化报错，注释掉运行正常
	# image = tf.image.per_image_standardization(image)   #标准化

	image_batch, label_batch = tf.train.batch([image, label],  # 生成批次
	                                          batch_size = batch_size,
	                                          num_threads = 64,
	                                          capacity = capacity)

	# you can also use shuffle_batch
	#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
	#                                                      batch_size=BATCH_SIZE,
	#                                                      num_threads=64,
	#                                                      capacity=CAPACITY,
	#                                                      min_after_dequeue=CAPACITY-1)

	# 这一步多余，删除无影响
	# label_batch = tf.reshape(label_batch, [batch_size])

	return image_batch, label_batch


def test():
	BATCH_SIZE = 2
	CAPACITY = 256
	IMG_W = 208
	IMG_H = 208

	train_dir = r'F:\DataResposity\dogs-vs-cats\train'
	image_list, label_list = get_files(train_dir)  # 读取数据和标签
	image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)  # 将图片分批次

	with tf.Session() as sess:
		i = 0
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord = coord)

		try:
			while not coord.should_stop() and i < 1:

				img, label = sess.run([image_batch, label_batch])

				# just test one batch
				for j in np.arange(BATCH_SIZE):
					print('label: %d' % label[j])  # j-index of quene of Batch_size
					plt.imshow(img[j, :, :, :])
					plt.show()
				i += 1

		except tf.errors.OutOfRangeError:
			print('done!')
		finally:
			coord.request_stop()
		coord.join(threads)


test()
