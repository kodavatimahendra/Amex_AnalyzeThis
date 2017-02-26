"""
This is based on MNIST tutorial in tensorflow website
This memorizes ~70% of the training data which is (I don't know, probably bad).
But hopefully a good starting point
"""

import csv
import numpy as np 
from decode_strings import char_party, index_party
from decode_strings import char_age, index_age
from decode_strings import char_education, index_education
from decode_strings import char_target

import tensorflow as tf
from tensorflow.contrib import rnn


train_data_file = "Datasets/Training_Dataset.csv"
test_data_file = "Datasets/Leaderboard_Dataset.csv"
train_data_partial_file = "Datasets/Twist_final_data.csv"
test_data_2_file = "Datasets/Final_Dataset.csv"

#Parameters
learning_rate = 3e-3
batch_size = 128
display_step = 100

#Network Parameters
num_input = 32 # Data shape
num_hidden = 128 # hidden layer num of features
num_classes = 5 # 5 parties: Centaur, Cosmos, Ebony, Odyssey, Tokugawa
num_examples = 20000
num_batches = 20000
num_batches_per_epoch = int(num_examples/batch_size)

x = []
y1 = []
y2 = []
with open(train_data_file,'rb') as csvfile:
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

train_inputs = np.asarray(y1)	
train_targets = np.asarray(y2)


X = tf.placeholder(tf.float32,[None, num_input])
w = tf.Variable(tf.truncated_normal([num_input, num_classes],stddev=0.1))
b = tf.Variable(tf.constant(0.1,shape=[num_classes]))

Y = tf.nn.relu(tf.matmul(X,w)+b)
Y_ = tf.placeholder(tf.float32,[None,num_classes])
cross_entrophy = -tf.reduce_sum(Y_ * tf.log(Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entrophy)

is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for batch in range(num_batches):
	indexes = [i % num_examples for i in range(batch*batch_size, (batch+1)*batch_size)]

	batch_train_inputs = train_inputs[indexes,:]
	batch_train_targets = train_targets[indexes,:]
	
	train_data = {X:batch_train_inputs, Y_:batch_train_targets}
	sess.run(train_step, feed_dict=train_data)

	if batch%display_step == 0:
		print("step {}/{}".format(batch, num_examples))


a,c = sess.run([accuracy,cross_entrophy], feed_dict={X:train_inputs,Y_:train_targets})
print("accuracy : {}".format(a))
