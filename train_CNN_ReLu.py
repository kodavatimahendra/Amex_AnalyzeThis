"""
This is based on deep MNIST tutorial in tensorflow website
This gives a accuracy of ~80% on the training data,
but gives only ~40% on test data.
The network is memorizing well but not generalizing well.
Probably overfitting or the maxpooling is wrong or reshaping of input vector is wrong.
Most probably CNN's do not work well with this because \
the adjacent data points are correleated but not like the way they are correleated in a image
"""

import csv
import numpy as np 
from decode_strings import char_party, index_party
from decode_strings import char_age, index_age
from decode_strings import char_education, index_education
from decode_strings import char_target

import tensorflow as tf

train_data_file = "Datasets/TrainDatasetFrom_Training_Dataset.csv"
test_data_file = "Datasets/TestDatasetFrom_Training_Dataset.csv"

#Some Parameters
batch_size = 128
display_step = 100

#Network Parameters 
num_input = 32 # Data shape
num_hidden = 128 # hidden layer num of features
num_classes = 5 # 5 parties: Centaur, Cosmos, Ebony, Odyssey, Tokugawa
num_examples = 48102
num_batches = 5000

def readingDataset(input_file):
	x = []
	y1 = []
	y2 = []
	with open(input_file,'rb') as csvfile:
		csv_variable = csv.reader(csvfile, delimiter="," , quotechar="|")
		next(csv_variable,None)                ##To skip the header in csv file
		for i,row in enumerate(csv_variable):
			x = row
			if x[31] == '':                     #I have a slightly corrupted dataset :-(
				pass
			else:			
				x[1] = char_party[x[1]]      	#decoding party_last_voted
				x[28] = char_age[x[28]]      	#decoding age
				x[31] = char_education[x[31]]   #decoding education
				y1.append([int(i) for i in x[1:33]])
				y2.append(char_target[x[35]])
	return (y1,y2)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1],
						strides=[1,2,2,1], padding='SAME')


x  = tf.placeholder(tf.float32, shape=[None, num_input])
y_ = tf.placeholder(tf.float32, shape=[None, num_classes])
x_image = tf.reshape(x, [-1,8,4,1])

W_conv1 = weight_variable([3,3,1,32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3,3,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([2 * 1 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1,2*1*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024,5])
b_fc2 = bias_variable([5])
y_conv = tf.matmul(h_fc1_drop, W_fc2)+b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


train_inputs, train_targets = readingDataset(train_data_file)
train_targets = np.asarray(train_targets)
train_inputs = np.asarray(train_inputs)
test_inputs ,test_targets = readingDataset(test_data_file)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for batch in range(num_batches):
	indexes = [i%num_examples for i in range(batch*batch_size, (batch+1)*batch_size)]

	batch_train_inputs = train_inputs[indexes,:]
	batch_train_targets = train_targets[indexes,:]

	train_data = {x:batch_train_inputs, y_:batch_train_targets, keep_prob:0.5}
	sess.run(train_step, feed_dict=train_data)


	if batch%display_step == 0:
		a = (accuracy.eval(feed_dict={x:train_inputs, y_:train_targets, keep_prob:1.0}))
		print("step {}/ {} training_accuracy {}".format(batch, num_examples,a))


print("Final Training accuracy")
print(accuracy.eval(feed_dict={x:test_inputs, y_:test_targets, keep_prob:1.0}))