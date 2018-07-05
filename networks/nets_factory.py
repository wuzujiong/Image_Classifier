"""Contains a factory for building various models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf
from networks import vgg_model
from networks import cifarnet
from networks import resnet_model_v2, resnet_v2
from tensorflow.contrib import slim

networks_map = {'vgg_16': vgg_model.vgg_16,
				'cifarnet': cifarnet.cifarnet,
                'resnet': resnet_v2.resnet_v2_50}

arg_scopes_map = {'vgg_16': vgg_model.vgg_arg_scope,
				  'cifarnet': cifarnet.cifarnet_arg_scope,
                  'resnet': resnet_v2.resnet_arg_scope}

def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
	"""Returns a network_fn such as `logits, end_points = network_fn(images)`.

	Args:
	  name: The name of the network.
	  num_classes: The number of classes to use for classification. If 0 or None,
	    the logits layer is omitted and its input features are returned instead.
	  weight_decay: The l2 coefficient for the model weights.
	  is_training: `True` if the model is being used for training and `False`
	    otherwise.

	Returns:
	  network_fn: A function that applies the model to a batch of images. It has
	    the following signature:
	        net, end_points = network_fn(images)
	    The `images` input is a tensor of shape [batch_size, height, width, 3]
	    with height = width = network_fn.default_image_size. (The permissibility
	    and treatment of other sizes depends on the network_fn.)
	    The returned `end_points` are a dictionary of intermediate activations.
	    The returned `net` is the topmost layer, depending on `num_classes`:
	    If `num_classes` was a non-zero integer, `net` is a logits tensor
	    of shape [batch_size, num_classes].
	    If `num_classes` was 0 or `None`, `net` is a tensor with the input
	    to the logits layer of shape [batch_size, 1, 1, num_features] or
	    [batch_size, num_features]. Dropout has not been applied to this
	    (even if the network's original classification does); it remains for
	    the caller to do this or not.

	Raises:
	  ValueError: If network `name` is not recognized.
	"""
	if name not in networks_map:
 		raise ValueError('Name of network unknow %s' % name)
	func = networks_map[name]
	@functools.wraps(func)
	def network_fn(images, **kwargs):
		arg_scopes = arg_scopes_map[name](weight_decay=weight_decay)
		with slim.arg_scope(arg_scopes):
			return func(images, num_classes, is_training=is_training, **kwargs)
	if hasattr(func, 'defaul_image_size'):
		network_fn.default_image_size = func.defaul_image_size
	return network_fn

