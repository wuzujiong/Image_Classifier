'''Contains model definitions for versions of VGG network'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim



def vgg_arg_scope(weight_decay=0.005):
	''' Define the VGG arg scope.
	Args:
		weight_decay: The l2 regularization coefficient.
	Return:
		A arg_scope
	'''
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
	                    activation_fn=tf.nn.relu,
	                    weights_regularizer=slim.l2_regularizer(weight_decay),
	                    biases_initializer=tf.zeros_initializer):
		with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
			return arg_sc


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
	with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
		end_points_collection = sc.original_name_scope + '_end_points'
		with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
		                    outputs_collections=end_points_collection):
			net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3,], scope='conv1')
			net = slim.max_pool2d(net, [2, 2,], scope='pool1')
			net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
			net = slim.max_pool2d(net, [2, 2], scope='pool2')
			net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
			net = slim.max_pool2d(net, [2, 2], scope='pool3')
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
			net = slim.max_pool2d(net, [2, 2], scope='pool4')
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
			net = slim.max_pool2d(net, [2, 2], scope='pool5')

			# Use conv2d instead of fully connected layers
			net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
			net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
			net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
			# Convert end_points_collection to a end_point_dict
			end_points = slim.utils.conver_collection_to_dict(end_points_collection)

			if global_pool:
				net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
				end_points['global_pool'] = net
			if num_classes:
				net = slim.dropout(net,
				                   dropout_keep_prob,
				                   is_training=is_training,
				                   scope='dropout7')
				net = slim.conv2d(net, num_classes, [1, 1],
				                  activation_fn=None,
				                  normalizer_fn=None,
				                  scope='fc8')
				if spatial_squeeze:
					net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
					end_points[sc.name + '/fc8'] = net
			return net, end_points

vgg_16.default_image_size = 224