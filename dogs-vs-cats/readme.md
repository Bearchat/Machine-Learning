https://blog.csdn.net/qq_16137569/article/details/72802387

data文件夹下包含test和train两个子文件夹，分别用于存放测试数据和训练数据，从官网上下载的数据直接解压到相应的文件夹下即可
logs文件夹用于存放我们训练时的模型结构以及训练参数
input_data.py负责实现读取数据，生成批次（batch）
model.py负责实现我们的神经网络模型
training.py负责实现模型的训练以及评估

分成数据读取、模型构造、模型训练、测试模型