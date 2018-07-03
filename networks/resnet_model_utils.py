from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import collections
import tensorflow as tf
from tensorflow.contrib import slim


class Block(collections.namedtuple('Bclock', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """

def subsample(inputs, factor, scope):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

def conv2d_same(inputs, num_outputs, kernel_size, rate=1, stride=1, scope=None):
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, rate=rate, stride=1, padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           rate=rate, padding='VALID', scope=scope)

def stack_block_dense(net, blocks, output_stride=None,
                      store_non_strided_activation=False,
                      output_collections=None):
    current_stride = 1
    rate = 1

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]):
            block_stride = 1
            for i, unit in enumerate(block.args):
                block_stride = unit.get('stride', 1)
