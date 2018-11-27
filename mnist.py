# -*- coding: utf-8 -*-
import sys
import os
import struct
import numpy as np
import matplotlib as plt
import tensorflow as tf

#导入input_data用于自动下载和安装MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

#创建两个占位符，x为输入网络的图像，y_为输入网络的图像类别
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#权重初始化函数
def weight_variable(shape):
    #输出服从截尾正态分布的随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#偏置初始化函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#创建卷积op
#x 是一个4维张量，shape为[batch,height,width,channels]
#卷积核移动步长为1。填充类型为SAME,可以不丢弃任何像素点, VALID丢弃边缘像素点
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

#创建池化op
#采用最大池化，也就是取窗口中的最大值作为结果
#x 是一个4维张量，shape为[batch,height,width,channels]
#ksize表示pool窗口大小为2x2,也就是高2，宽2
#strides，表示在height和width维度上的步长都为2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding="SAME")

#第1层，卷积层
#初始化W为[5,5,1,6]的张量，表示卷积核大小为5*5，1表示图像通道数，6表示卷积核个数即输出6个特征图
W_conv1 = weight_variable([5,5,1,6])
#初始化b为[6],即输出大小
b_conv1 = bias_variable([6])

#把输入x(二维张量,shape为[batch, 784])变成4d的x_image，x_image的shape应该是[batch,28,28,1]
#-1表示自动推测这个维度的size
x_image = tf.reshape(x, [-1,28,28,1])

#把x_image和权重进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max_pooling
#h_pool1的输出即为第一层网络输出，shape为[batch,14,14,6]
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第2层，卷积层
#卷积核大小依然是5*5，通道数为6，卷积核个数为16
W_conv2 = weight_variable([5,5,6,16])
b_conv2 = weight_variable([16])

#h_pool2即为第二层网络输出，shape为[batch,7,7,16]
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#第3层, 全连接层
#这层是拥有1024个神经元的全连接层
#W的第1维size为7*7*16，7*7是h_pool2输出的size，16是第2层输出神经元个数
W_fc1 = weight_variable([7*7*16, 120])
b_fc1 = bias_variable([120])

#计算前需要把第2层的输出reshape成[batch, 7*7*16]的张量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout层
#为了减少过拟合，在输出层前加入dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
#最后，添加一个softmax层
#可以理解为另一个全连接层，只不过输出时使用softmax将网络输出值转换成了概率
W_fc2 = weight_variable([120, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#预测值和真实值之间的交叉墒
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

#train op, 使用ADAM优化器来做梯度下降。学习率为0.0001
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。
#因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
correct_predict = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

#计算正确预测项的比例，因为tf.equal返回的是布尔值，
#使用tf.cast把布尔值转换成浮点数，然后用tf.reduce_mean求平均值
accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))

saver = tf.train.Saver()

#打印mnist数组(一个数字)，以数字形式显示
def print_digit(digit_array):
    array = digit_array
    array[array>0] = 1
    digitint = array.astype(int)
    digit = digitint.reshape(28,28)
    for i in range(28):
        for j in range(28):
            print(digit[i][j], end='')
        print('')

#开始训练模型，循环20000次，每次随机从训练集中抓取50幅图像
def cnn_train():
    # 创建一个交互式Session
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            #每100次输出一次日志
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_:batch[1], keep_prob:1.0})
            print ("step %d, training accuracy %g" % (i, train_accuracy))
            saver.save(sess, './model/minst')
        train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
    sess.close()

#对测试集进行验证
def mnist_test():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, 'model/minst')
    print( "测试集验证结果:%g" % accuracy.eval(feed_dict={
        x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
    sess.close()

#mnist测试集中取10个样本进行预测
def predict_mnist_array():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, 'model/minst')

    test_ds = mnist.test.images
    for i in range(10):
        print("----------------------------")
        print("第%d个手写数字矩阵形状:"%i)
        print_digit(test_ds[i])
        predict = sess.run(y_conv, feed_dict = {x: [test_ds[i]], y_: [mnist.test.labels[i]], keep_prob:0.5})
        print("预测值:", sess.run(tf.argmax(predict[0])))
        print("实际值:", sess.run(tf.argmax(mnist.test.labels[i])))
        print("")
    sess.close()

#检查文件是否为bitmap格式
def check_is_bitmap_file(file_path):  
    with open(file_path, 'rb') as f:
        #将读取到的30个字节，转换为指定数据类型的数字
        head = struct.unpack('<ccIIIIIIHH', f.read(30)) 
        #print(head)
        if head[0] == b'B' and head[1] == b'M':
            #print('%s 总大小：%d, 图片尺寸：%d X %d, 颜色数：%d' % (file, head[2], head[6], head[7], head[9]))
            #返回图片总大小，以及一个像素用多少bit表示
            return True, head[2], head[9] 
        else:
            #print('%s 不是Bmp图片' % file)
            return False, 0, 0

#读取bitmap文件，返回[1][784]形式数组(将bmp转换成mnist数据集格式，方便预测)
def read_28x28x8_bitmap(file_path):
    while True:
        if not os.path.isfile(file_path):
            print("传入的文件名不合法,请传入一个bitmap格式的文件.")
            break
        ret, file_size, bits = check_is_bitmap_file(file_path)
        if not ret:
            print("非法的bitmap文件")
            break
        if bits != 8:
            print("非法的bitmap文件,只支持28*28*8格式")
            break
        print("file_size=%d, bits=%d" % (file_size, bits))
        with open(file_path, 'rb') as f:
            formate = '%dB' % file_size
            data_bytes = struct.unpack(formate, f.read(file_size))
            image_array = np.zeros((1, 784))
            step = bits / 8
            #54字节文件头，1024字节调色板
            start_pos = 54 + 1024
            #读取28*28像素, bmp文件像素从左到右，从下到上
            for i in range(28):
                for j in range(28):
                    pos = int(start_pos + (28 - i - 1) * 28 + j * step) 
                    pixel_value = data_bytes[pos] / 255.0
                    image_array[0][i*28 + j] = pixel_value
        return True, image_array
    #错误返回
    return False, []

#对一个bitmap文件进行预测
def predict_bitmap(file_path):
    #读取bitmap文件
    ret, data_array = read_28x28x8_bitmap(file_path)
    if not ret:
        return

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, 'model/minst')
    print("----------------------------")
    print("手写数字矩阵形状:")
    print_digit(data_array)
    predict = sess.run(y_conv, feed_dict = {x: data_array, y_: [mnist.test.labels[0]], keep_prob:0.5})
    print("预测值:", sess.run(tf.argmax(predict[0])))
    sess.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\n请传入一个参数,指定要进行的操作, 下面是参数说明:")
        print("train参数,用于训练模型")
        print("test参数,用测试数据集评估准确度")
        print("predict参数,用测试集前10个测试样本进行预测")
        print("传入bmp文件路径,用于对bmp文件进行预测,只支持28*28*8格式的位图")
        sys.exit(1)
    if sys.argv[1] == "train":
        cnn_train()
    elif sys.argv[1] == "test":
        mnist_test()
    elif sys.argv[1] == "predict":
        predict_mnist_array()
    else:
        predict_bitmap(sys.argv[1])
