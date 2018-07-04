r"""Downloads and converts cifar10 data to TFRecords of TF-Example protos.

This module downloads the cifar10 data, uncompresses it, reads the files
that make up the cifar10 data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take several minutes to run.

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

from datasets import dataset_utils

# The number of training files
_NUM_TRAIN_FILES = 5
# The height and width of image
_IMAGE_SIZE = 32

# The name of the classes
_CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

def _add_to_tfrecord(filename, tfrecord_writer, offset=0):
    """Loads data from the cifar10 pickle files and writes files to a TFRecord.

    Args:
      filename: The filename of the cifar10 pickle file.
      tfrecord_writer: The TFRecord writer to use for writing.
      offset: An offset into the absolute number of images previously written.

    Returns:
      The new offset.
    """
    with tf.gfile.Open(filename, 'rb') as f:
        if sys.version_info < (3,):
            data = cPickle.load(f)
        else:
            data = cPickle.load(f, encoding='bytes')
    images = data[b'data']
    num_images = images.shape[0]
    images = images.reshape((num_images, 3, 32, 32))
    labels =data[b'labels']

    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encode_image = tf.image.encode_png(image_placeholder)
        with tf.Session('') as sess:
            for j in range(num_images):
                sys.stdout.write('\r>> Reading file [%s} image %d/%d' %
                                 (filename, offset + j + 1, offset + num_images))
                sys.stdout.flush()

                image = np.squeeze(images[j]).transpose((1, 2, 0))
                label = labels[j]
                png_string = sess.run(encode_image,
                                      feed_dict={image_placeholder: image})
                example = dataset_utils.image_to_tfexample(
                    png_string, b'png', _IMAGE_SIZE, _IMAGE_SIZE, label)
                tfrecord_writer.write(example.SerializeToString())
    return offset + num_images

def _get_output_filename(dataset_dir, split_name):
  """Creates the output filename.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  return '%s/cifar10_%s.tfrecord' % (dataset_dir, split_name)

def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  training_filename = _get_output_filename(dataset_dir, 'train')
  testing_filename = _get_output_filename(dataset_dir, 'test')

  if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  # First, process the training data:
  with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
    offset = 0
    for i in range(_NUM_TRAIN_FILES):
      filename = os.path.join(dataset_dir,
                              'cifar-10-batches-py',
                              'data_batch_%d' % (i + 1))  # 1-indexed.
      offset = _add_to_tfrecord(filename, tfrecord_writer, offset)

  # Next, process the testing data:
  with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
    filename = os.path.join(dataset_dir,
                            'cifar-10-batches-py',
                            'test_batch')
    _add_to_tfrecord(filename, tfrecord_writer)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  print('\nFinished converting the Cifar10 dataset!')\

if __name__ == '__main__':
    dataset_dir = 'F:/DL_Datasets/cifar-10-python'
    run(dataset_dir=dataset_dir)