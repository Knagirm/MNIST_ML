from __future__ import print_function
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

def compute_accuracy(v_xs, v_ys):
	global prediction, loss
	y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
	correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})

	# loss = sess.run([loss], feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
	return result

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	# stride [1, x_movement, y_movement, 1]
	# Must have strides[0] = strides[3] = 1
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	# stride [1, x_movement, y_movement, 1]
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

dim_input = 784
dim_output = 10

max_epochs = 15
learn_rate = 1e-4
batch_size = 100


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])	

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, x[-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
											  reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

sess = tf.Session()

if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	init = tf.initialize_all_variables()
else:
	init = tf.global_variables_initializer()

sess.run(init)


xtrain = np.load('data/mnist/xtrain.npy')
ytrain = np.load('data/mnist/ytrain.npy')

xval = np.load('data/mnist/xval.npy')
yval = np.load('data/mnist/yval.npy')

xtest = np.load('data/mnist/xtest.npy')


train_arr = []
# train_loss = []
val_arr = []
# val_loss = []
index = []

num_steps = 500

for j in range(max_epochs):
	for i in range(num_steps):
		#batch_xs, batch_ys = mnist.train.next_batch(100)
		batch_xs = xtrain[i*batch_size:(i+1)*batch_size]
		batch_ys = ytrain[i*batch_size:(i+1)*batch_size]
		sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
		
		if (i % 50 == 0):
			acc = compute_accuracy(batch_xs, batch_ys)
			index.append(j*500+i)
			train_arr.append(acc*100)
			# train_loss.append(loss)

			acc = compute_accuracy(xval, yval)
			print(j, i, acc)
			val_arr.append(acc*100)
			# val_loss.append(loss)

plt.plot(index, train_arr)
plt.show()

plt.plot(index, val_arr)
plt.show()

# y_pre = sess.run(prediction, feed_dict={xs: xtest, keep_prob: 1})
# arr_pre = sess.run(tf.argmax(y_pre,1))
# np.savetxt('out.txt',arr_pre)