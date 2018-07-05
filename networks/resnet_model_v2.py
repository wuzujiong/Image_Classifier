from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib import slim
from networks import resnet_utils


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

def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None,
                               activation_fn=None,
                               scope='conv3')
        output = residual + shortcut
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

blocks_50 = [
    {'scope_name': 'block1', 'depth_bottleneck': 64, 'depth': 256, 'num_units': 3, 'stride': 2},
    {'scope_name': 'block2', 'depth_bottleneck': 128, 'depth': 512, 'num_units': 3, 'stride': 2},
    {'scope_name': 'block3', 'depth_bottleneck': 256, 'depth': 1024, 'num_units': 3, 'stride': 2},
    {'scope_name': 'block4', 'depth_bottleneck': 512, 'depth': 2048, 'num_units': 3, 'stride': 1}]

# def stack_block_dense(net, l)

def resenet50_v2(inputs, num_classes=None,
                 global_pool=False, spatial_squeeze=True,
                 is_training=True, scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs]) as sc:
        end_points_collection = sc.orignal_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs

                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

                # Structure the blocks
                rate = 1
                for block in blocks_50:
                    with tf.variable_scope(block['scope_name'], 'block', [net]) as sc:
                        block_stride = 1
                        for i in range(block['num_units']):
                            if i == block['num_units'] - 1:
                                block_stride = block['stride']
                            with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                                net = bottleneck(net,
                                                 block['depth'],
                                                 block['depth_bottleneck'],
                                                 stride=block_stride,
                                                 rate=rate)
                    net = slim.utils.collect_named_outputs(end_points_collection, sc.name, net)
                    net = subsample(net, block_stride)

                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                    end_points['global_pool'] = net

                if num_classes:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net

                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='Spatial_squeeze')
                    end_points[sc.name + '/spatial_squeeze'] = net

                end_points['predictions'] = slim.softmax(net, scope='predictions')

resenet50_v2.defaults_image_size = 224








