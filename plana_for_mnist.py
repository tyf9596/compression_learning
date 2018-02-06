# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:07:18 2017

@author: tyf
"""

#!/usr/bin/python3.5  
# -*- coding: utf-8 -*-    
  
import os  
  
import numpy as np   
import tensorflow as tf  
  
from PIL import Image  
from numpy import random
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
compression_nodenum=12 ##压缩感知节点数
compensate_nodenum=12###补偿节点数
total_nodenum=compression_nodenum+compensate_nodenum##输入训练网络节点数

obmatrix = 1 * np.random.randn(784,compression_nodenum) +0
#obmatrix2 = 1 * np.ones(784,compression_nodenum) +0
cinput=np.array([[0]*total_nodenum for i in range(55000)],dtype=np.float)
tinput=np.array([[0]*total_nodenum for i in range(10000)],dtype=np.float)
cbuchong_input=np.array([[0]*compensate_nodenum for i in range(55000)],dtype=np.float)
cbuchong_test=np.array([[0]*compensate_nodenum for i in range(10000)],dtype=np.float)
cbuchong_train=np.array([[0]*compensate_nodenum for i in range(55000)],dtype=np.float)
#cmat1=np.array([[0]*compression_nodenum for i in range(10000)],dtype=np.float)
print ("压缩率为")
print(total_nodenum/784)

#返回大的数值
def bidaxiao(a,b):
    if (a>=b):
        return a
    else:
        return b
    
#设置压缩参数
#压缩函数

def planA(input_images,cmat):
    print(cmat.shape)
    for i in range(1,input_images.shape[0]):
        temp=np.zeros((4,3))
        temp=cmat[i].reshape((4,3))
        imager=input_images[i].reshape(4,196)
        for j in range(4):
            maxkeep1=0
            maxkeep0=0
            changes=0
            temp_maxkeep0=0
            temp_maxkeep1=0
            for k in range(1 ,196):
                if (imager[j][k]==imager[j][k-1]):
                    if(imager[j][k]==0):
                        temp_maxkeep0=temp_maxkeep0+1
                        maxkeep0=bidaxiao(maxkeep0,temp_maxkeep0)
                    else:
                        temp_maxkeep1=temp_maxkeep1+1
                        maxkeep1=bidaxiao(maxkeep1,temp_maxkeep1)
                else:
                     temp_maxkeep0=0
                     temp_maxkeep1=0
                     changes=changes+1
            temp[j][0]=maxkeep0
            temp[j][1]=maxkeep1
            temp[j][2]=changes
        cmat[i]=temp.reshape(-1,12)
        
planA(mnist.test.images,cbuchong_test)        
print (cbuchong_test[1],cbuchong_test[2],cbuchong_test[3])
planA(mnist.train.images,cbuchong_train)


def cdataset(input_images,cmat,obmat,cbuchong):
    
    for i in range (1,input_images.shape[0]):
        #print(i)
        #print(np.matmul(input_images[i],obmat).shape)
        #print(cbuchong[i].shape)
        cmat[i]=np.column_stack((np.matmul(input_images[i],obmat).reshape(cbuchong[i].shape[0],-1),cbuchong[i])).reshape(total_nodenum)
        
       #cmat[i] = np.fft.fft(cmat[i])
    return (cmat)
#生成训练集和测试集
cinput=cdataset(mnist.train.images,cinput,obmatrix,cbuchong_train)
tinput=cdataset(mnist.test.images,tinput,obmatrix,cbuchong_test)
       
# 第一次遍历图片目录是为了获取图片总数  
input_count = 0 
''' 
for i in range(0,10):  
    dir = './mnist_digits_images/%s/' % i                 # 这里可以改成你自己的图片目录，i为分类标签  
    for rt, dirs, files in os.walk(dir):  
        for filename in files:  
            input_count += 1  
  
# 定义对应维数和各维长度的数组  
input_images = np.array([[0]*784 for i in range(input_count)],dtype=np.float)  
input_labels = np.array([[0]*10 for i in range(input_count)])  
#cinput=np.array([[0]*compression_nodenum for i in range(input_count)],dtype=np.float)
  
# 第二次遍历图片目录是为了生成图片数据和标签  
index = 0  
for i in range(0,10):  
    dir = './mnist_digits_images/%s/' % i                 # 这里可以改成你自己的图片目录，i为分类标签  
    for rt, dirs, files in os.walk(dir):  
        for filename in files:  
            filename = dir + filename  
            img = Image.open(filename)  
            width = img.size[0]  
            height = img.size[1]  
            for h in range(0, height):  
                for w in range(0, width):  
                    # 通过这样的处理，使数字的线条变细，有利于提高识别准确率  
                    if img.getpixel((w, h)) > 230:  
                        input_images[index][w+h*width] = 0  
                    else:  
                        input_images[index][w+h*width] = 1
            #以下内容为自己改的做压缩感知的部分
            input_images[index]=input_images[index].astype(np.float64)
            #print(input_images[index])
            #print(x_test)
            #print(obmatrix)
            input_images[index]=np.matmul(input_images[index],obmatrix)
            #print(input_images[index])
            #print(np.matmul(x_test,obmatrix))
            #yuandaima           
            input_labels[index][i] = 1  
            index += 1  
  
'''  
# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)  
x = tf.placeholder(tf.float32, shape=[None, total_nodenum])  
y_ = tf.placeholder(tf.float32, shape=[None, 10])  
  
x_image = tf.reshape(x, [-1, 4, 6, 1])  
#print(x_image)  
# 定义第一个卷积层的variables和ops  
W_conv1 = tf.Variable(tf.truncated_normal([4, 4, 1, 32], stddev=0.1))  
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))  
  
L1_conv = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')  
L1_relu = tf.nn.relu(L1_conv + b_conv1)  
L1_pool = tf.nn.max_pool(L1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
  
# 定义第二个卷积层的variables和ops  
W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))  
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))  
  
L2_conv = tf.nn.conv2d(L1_pool, W_conv2, strides=[1, 1, 1, 1], padding='SAME')  
L2_relu = tf.nn.relu(L2_conv + b_conv2)  
L2_pool = tf.nn.max_pool(L2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
  
  
# 全连接层  
W_fc1 = tf.Variable(tf.truncated_normal([1 * 2 * 64,1024], stddev=0.1))  
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))  
  
h_pool2_flat = tf.reshape(L2_pool, [-1, 1*2*64])  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  
  
  
# dropout  
keep_prob = tf.placeholder(tf.float32)  
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  
  
  
# readout层  
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))  
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))  
  
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  
  
# 定义优化器和训练op  
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  
train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)  
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
#代码持久化
saver=tf.train.Saver()
  
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
  
    print ("一共读取了 %s 个输入图像， %s 个标签" % (input_count, input_count))  
  
    # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）  
    input_count=55000
    batch_size = 50  
    iterations = 500  
    batches_count = int(input_count / batch_size)  
    remainder = input_count % batch_size  
    print ("数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count+1, batch_size, remainder))  
  
    # 执行训练迭代  
    for it in range(iterations):  
        # 这里的关键是要把输入数组转为np.array  
        for n in range(batches_count):  
            train_step.run(feed_dict={x: cinput[n*batch_size:(n+1)*batch_size], y_: mnist.train.labels[n*batch_size:(n+1)*batch_size], keep_prob: 0.5})  
        if remainder > 0:  
            start_index = batches_count * batch_size;  
            train_step.run(feed_dict={x: cinput[start_index:input_count-1], y_: mnist.train.labels[start_index:input_count-1], keep_prob: 0.5})  
  
        # 每完成五次迭代，判断准确度是否已达到100%，达到则退出迭代循环  
        iterate_accuracy = 0  
        if it%5 == 0:  
            iterate_accuracy = accuracy.eval(feed_dict={x: tinput, y_: mnist.test.labels, keep_prob: 1.0})  
            print ('iteration %d:in test_dataset accuracy  %s' % (it, iterate_accuracy))  
            #train_iterate_accuracy = accuracy.eval(feed_dict={x: cinput, y_: mnist.train.labels, keep_prob: 1.0})
            #print ('iteration  %d: in train_datasetaccuracy %s' % (it, train_iterate_accuracy)) 
            if iterate_accuracy >= 1:  
                break;  
    saver.save(sess,"D:/Fudan/compressed_learning/minst/model/model625fft.ckpt")
    print ('完成训练!')  
