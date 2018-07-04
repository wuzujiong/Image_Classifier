"""
The script is conver cats_vs_dogs to tfrecord format dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

import numpy as np
from six.moves import cPickle
from six.moves import urllib
import tensorflow as tf
from tensorflow.contrib import slim
import cv2


from datasets import dataset_utils

# dataset_dir = 'F:/DL_Datasets/Dogs vs. Cats Redux/Dogs vs. Cats Redux_Images'
dataset_dir = 'F:/DL_Datasets/Dogs vs. Cats Redux/images'

tfrecord_save_dir = 'F:/DL_Datasets/Dogs vs. Cats Redux/Dogs vs. Cats Redux_tfrecorder'

_NUM_TRAIN = 20000
_NUM_TEST = 5000

def _get_output_filename(dataset_dir, split_name):
  """Creates the output filename.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  return '%s/CatVSDog_%s.tfrecord' % (dataset_dir, split_name)

def _add_to_tfrecord(filename, tfrecord_writer):

    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dir, image_name = os.path.split(filename)
    start_index = len(dir) + 1
    if filename[start_index:start_index + 3] == 'cat':
        label = int(0)
    elif filename[start_index:start_index + 3] == 'dog':
        label = int(1)
    else:
        raise ValueError('Label Error for classes')

    shape = image.shape
    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encode_image = tf.image.encode_png(image_placeholder)
        with tf.Session('') as sess:
            png_string = sess.run(encode_image, feed_dict={image_placeholder: image})
            example = dataset_utils.image_to_tfexample(
                png_string, b'png', shape[0], shape[1], label)
            tfrecord_writer.write(example.SerializeToString())

def run(dataset_dir, tfrecord_save_dir):

    if not tf.gfile.Exists(dataset_dir):
        raise ValueError('dataset_dir path is empty, you must input a correct path.')

    if not tf.gfile.Exists(tfrecord_save_dir):
        tf.gfile.MakeDirs(tfrecord_save_dir)
    training_filename = _get_output_filename(tfrecord_save_dir, 'train')
    testing_filename = _get_output_filename(tfrecord_save_dir, 'test')

    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    all_filenames = os.listdir(dataset_dir)
    np.random.shuffle(all_filenames)

    tfrecord_writer = tf.python_io.TFRecordWriter(training_filename)
    for i, image_name in enumerate(all_filenames):
        filename = os.path.join(dataset_dir, image_name)
        if i == _NUM_TRAIN:
            tfrecord_writer = tf.python_io.TFRecordWriter(testing_filename)
        _add_to_tfrecord(filename, tfrecord_writer)
        sys.stdout.write('r\>> Reading file [%s] image %d/%d' %
                         (filename, i, len(all_filenames)))
        sys.stdout.flush()
def get_image():
    dataset = slim.dataset.Dataset(
        data_sources=
    )

if __name__ == '__main__':
    run(dataset_dir, tfrecord_save_dir)

