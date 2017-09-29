from __future__ import division, print_function, absolute_import

import datetime
import random
# import ipdb
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import bitarray

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from PIL import Image

# images are grayscale
# size of individual images: 320 pixels (width) X 160 pixels (height)
# each input will consist of 3 stacked images - shape: 320 x 160 x 3

class train_autoencoder:
	def get_time(self):
		return "the time is " + str(datetime.datetime.now().time())[0:5]

	def start(self, amt_trn_data, file_num, num_trn_epochs, encodings, nn, batch_size, lr):

		tf.reset_default_graph()
		tf.logging.set_verbosity(tf.logging.INFO)

		print("starting...")

		# Create training, validation and testing sets
		amt_tr = amt_trn_data
		amt_v = 500
		amt_te = 2000
		# batch_size = 50

		train_inds = []
		valid_inds = []
		test_inds = []

		arr = range(12000)
		random.shuffle(arr)

		train_inds = arr[0 : amt_tr]
		valid_inds = arr[amt_tr : amt_tr + amt_v]
		test_inds = arr[amt_tr + amt_v : amt_tr + amt_v + amt_te]

		# Parameters
		learning_rate = lr # 0.01
		display_step = 1

		imgs = tf.placeholder(tf.float32, shape=(None, 160, 320, 3)) 

		train_imgs_summ = tf.summary.image('train-images', imgs, 10)
		test_imgs_summ = tf.summary.image('test-images', imgs, 10)
		valid_imgs_summ = tf.summary.image('valid-images', imgs, 10)

		output_imgs = self.autoencoder(imgs, encodings)

		train_output_imgs_summ = tf.summary.image('train-output-images', output_imgs, 10)
		valid_output_imgs_summ = tf.summary.image('validation-output-images', output_imgs, 10)
		test_output_imgs_summ = tf.summary.image('test-output-images', output_imgs, 10)

		# validation_tracker = [100, 0] # [lowest value, num_steps it's been the lowest value]
		# stopping_point = 400 

		# Define loss and optimizer, minimize the squared error
		cost = tf.reduce_mean(tf.pow(imgs - output_imgs, 2))

		# minimize training cost
		rmsprop = tf.train.RMSPropOptimizer(learning_rate)
		optimizer = rmsprop.minimize(cost)

		train_loss_summ = tf.summary.scalar("training-loss", cost)
		valid_loss_summ = tf.summary.scalar("validation-loss", cost)
		test_loss_summ = tf.summary.scalar("test-loss", cost)

		training_summ = tf.summary.merge([train_imgs_summ, train_output_imgs_summ, train_loss_summ])
		valid_summ = tf.summary.merge([valid_imgs_summ, valid_output_imgs_summ, valid_loss_summ])
		testing_summ = tf.summary.merge([test_imgs_summ, test_output_imgs_summ, test_loss_summ])

		# Initializing the variables
		init = tf.global_variables_initializer()

		print("launching the graph...")

		f = open("/root/TensorBoard/tr_%d_nn_%d_lr_%s/f%d.txt"
			% (amt_tr, nn, str(learning_rate)[2:], file_num), "w")

		# Launch the graph
		with tf.Session() as sess:
			sess.run(init)
					
			print("starting training cycle...")
			writer = tf.summary.FileWriter("/root/TensorBoard/tr_%d_nn_%d_lr_%s/%d"
				% (amt_tr, nn, str(learning_rate)[2:], file_num) )
			writer.add_graph(sess.graph)

			# Training cycle
			num_epochs = num_trn_epochs # 57 epochs for 500 training images
			for epoch in range(num_epochs):
				# Loop over all batches
					
				num_training_batches = int(amt_tr/batch_size)	
				for i in range(num_training_batches):
					# get training batch data (images)
					s = i * batch_size 
					e = s + batch_size
					batch = self.get_data(s, e, train_inds)

					# collect training summary data
					if i%10==0:
						step = epoch * num_training_batches + i
						train_summ_data = sess.run(training_summ, feed_dict={imgs: batch})
						writer.add_summary(train_summ_data, global_step = step)

					# Run optimization op (backprop) and cost op (to get loss value)
					_, train_loss = sess.run([optimizer, cost], feed_dict={imgs: batch})

				# get validation batch data
				num_validation_batches = int(amt_v/batch_size)
				valid_loss = 0 # validation loss

				for i in range(num_validation_batches):
					# collect validation summary data
					s = i * batch_size
					e = s + batch_size
					batch = self.get_data(s, e, valid_inds)

					if i%10==0:
						step = epoch * num_validation_batches + i
						valid_summ_data = sess.run(valid_summ, feed_dict={imgs: batch})
						writer.add_summary(valid_summ_data, global_step = step)
						
					# Get validation loss
					valid_loss += cost.eval(feed_dict={imgs: batch})		

				# Display logs per epoch step
				if epoch % display_step == 0:
					# print epoch # and cost to 9 decimal places
					prst = ("Epoch %04d, training loss: " % (epoch + 1)) + "{:.9f}".format(train_loss)
					f.write(prst)
					f.write("\n")

				# print (average) validation loss
				if epoch % display_step == 0:
					vl = valid_loss/(num_validation_batches*1.0)
					prst = "validation loss: " + str(vl)
					f.write(prst)
					f.write("\n")
					f.write(self.get_time())
					f.write("\n")
					f.write("\n")
				
				'''
				if vl < validation_tracker[0]:
					validation_tracker[0] = vl
					validation_tracker[1] = 0
				else:
					validation_tracker[1] += num_training_batches
					if validation_tracker[1] > stopping_point:
						break
				'''	

			print("Optimization Finished!")

			# Check accuracy
			test_loss = 0
			num_testing_batches = int(amt_te/batch_size)
			for i in range(num_testing_batches): 
				s = i * batch_size
				e = s + batch_size
				batch = self.get_data(s, e, test_inds)
						
				# collect testing summary data
				if i%10==0:
					test_summ_data = sess.run(testing_summ, feed_dict={imgs: batch})
					writer.add_summary(test_summ_data, global_step = i)
						
				test_loss += cost.eval(feed_dict={imgs: batch})
			prst = "test loss:" + str(test_loss/(num_testing_batches*1.0))
			f.write(prst)
			f.write("\n")
		f.close()

	def conv_layer(self, inputs, info, num):
		filters, kernel_size = info
		return tf.layers.conv2d (inputs=inputs, filters=filters, kernel_size=kernel_size,
					padding="same", activation=tf.nn.relu, name="conv{}".format(num) )

	def pool_layer(self, inputs, info, num):
		pool_size, strides = info
		return tf.layers.max_pooling2d (inputs=inputs, pool_size=pool_size, 
			strides=strides, name="pool{}".format(num) )

	def unpool_layer(self, inputs, size):
		return tf.image.resize_images(inputs, size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	def trans_conv_layer(self, inputs, info, num):
		filters, kernel_size = info
		return tf.layers.conv2d_transpose (inputs=inputs, filters=filters, kernel_size=kernel_size, 
			padding="same", activation=tf.nn.relu, name="trans-conv{}".format(num) )

	def batch_norm_layer(self, inputs):
		self.bnn += 1
		return tf.layers.batch_normalization(inputs, name="batch_normalization_{}".format(self.bnn))

	def dropout_layer(self, inputs, num):
		return tf.layers.dropout(inputs, rate=0.2, name="dropout{}".format(num))

	def autoencoder(self, input_layer, encodings):
		'''
		encodings: ( (conv_1_info, pool_1_info), ... , (conv_n_info, pool_n_info) )
		conv_info = (num_filters, kernel_size)
		pool_info = (pool_size, strides)

		deconv_info = ((num_filters, kernel_size), ... )
		unpool_info = (size, ... )
		'''
		print("entering autoencoder function...")
		self.bnn = -1 # batch normalization layer number
		shape = list(input_layer.get_shape())[1:]	
		shape = [int(s) for s in shape]

		N = len(encodings)
		encodings = encodings + (((3, None), ()),)
		deconv_info = tuple( (encodings[index+1][0][0], encodings[index][0][1]) 
			for index in range(N-1,-1,-1) )
		unpool_info = (shape[0:2],)
	
		for i in range(N):
			self.batch_norm_i = self.batch_norm_layer(input_layer)
			self.conv_i = self.conv_layer(self.batch_norm_i, encodings[i][0], i)
			self.pool_i = self.pool_layer(self.conv_i, encodings[i][1], i)
			shape = list(self.pool_i.get_shape())[1:]
			shape = [int(s) for s in shape] 

			unpool_info = (shape[0:2],) + unpool_info
			input_layer = self.pool_i

		unpool_info = unpool_info[1:]
		dim = list(input_layer.get_shape()[1:])
		dim = [int(d) for d in dim]
		dim_prod = np.prod(dim)

		self.pool_flat = tf.reshape(input_layer, [-1, dim_prod], name="flatten")

		input_layer = self.pool_flat

		for i,units in enumerate([256,64,256,dim_prod]):
			self.batch_norm_i = self.batch_norm_layer(input_layer)
			self.dense_i = tf.layers.dense(inputs=self.batch_norm_i, units=units, activation=tf.nn.relu, name="dense{}".format(i))	
			self.dropout_i = self.dropout_layer(self.dense_i, i)
			input_layer = self.dropout_i

		# unflatten layer: [-1, 80*40*3] => [-1, 40, 80, 3]
		self.unflatten = tf.reshape(input_layer, [-1] + dim, name="unflatten")

		input_layer = self.unflatten

		for i in range(N):
			self.unpool_i = self.unpool_layer(input_layer, unpool_info[i])
			self.batch_norm_i = self.batch_norm_layer(self.unpool_i)
			self.trans_conv_i = self.trans_conv_layer(self.batch_norm_i, deconv_info[i], i)
			input_layer = self.trans_conv_i
	
		print("exiting autoencoder function...")
		return input_layer

	def get_data(self,s,e,ind_arr):
		data = []
		fn = "/root/pongv0/DATA/%05d.bin"
			
		for img in range(s, e):
			bits = bitarray.bitarray()  
			i = 0
			
			with open(fn%(ind_arr[img]), 'rb') as fh:
				bits.fromfile(fh)

			img_array = []
			for y in range(160):
				row = []
				for x in range(320):
					layer = []
					for d in range(3):
						layer.append( int( bits[i] ) )
						i += 1
					row.append(layer)
				img_array.append(row)
			data.append(img_array)	
		return data
			
