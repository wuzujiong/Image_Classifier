from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib import slim
# from datasets import dataset_utils
# import matplotlib.pyplot as plt
from preprocessing_image import preprocessing_factory

_FILE_PATTERN = 'CatVSDog_%s.tfrecord'
_SPLITS_TO_SIZES = {'train': 20000, 'test': 5000}
# _DATASET_DIR = '/Users/prmeasure/Desktop/Dogs vs. Cats Redux/Dogs vs. Cats Redux_tfrecorder' # MAC
_DATASET_DIR = 'F:\DL_Datasets\Dogs vs. Cats Redux\Dogs vs. Cats Redux_tfrecorder' # WINDOWS

CATVSDOG_LABELS = {
	'cat': (0, 'cat'),
	'dog': (1, 'dog')
}

_ITEMS_TO_DESCRIPTIONS = {
	'image': 'A color image',
	'label': 'A single intergr between 0 and 1'
}

_SPLITE_NAME = 'train'
_BATCH_SIZE = 64
_NUM_CLASSES = 2
_IMAGE_SIZE = 229


def _get_split(split_name, dataset_dir, file_pattern, reader,
			  split_to_size, items_to_descriptions, num_classes):
	"""Gets a dataset tuple with instructions for reading Pascal VOC dataset.

	 Args:
	   split_name: A train/test split name.
	   dataset_dir: The base directory of the dataset sources.
	   file_pattern: The file pattern to use when matching the dataset sources.
		 It is assumed that the pattern contains a '%s' string so that the split
		 name can be inserted.
	   reader: The TensorFlow reader type.

	 Returns:
	   A `Dataset` namedtuple.

	 Raises:
		 ValueError: if `split_name` is not a valid train/test split.
	 """
	if split_name not in split_to_size:
		raise ValueError('split name %s was not recognized.' % split_name)
	file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
	if reader is None:
		reader = tf.TFRecordReader
	keys_to_features = {
		'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
		'image/format': tf.FixedLenFeature((), tf.string, default_value='JPEG'),
		'image/classes/label': tf.FixedLenFeature([1], dtype=tf.int64),
	}

	items_to_handlers = {
		'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
		'label': slim.tfexample_decoder.Tensor('image/classes/label'),
	}

	decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
													  items_to_handlers)
	label_to_names = None

	return slim.dataset.Dataset(
		data_sources=file_pattern,
		reader=reader,
		decoder=decoder,
		num_samples=split_to_size[split_name],
		items_to_descriptions=items_to_descriptions,
		num_classes=num_classes,
		label_to_names=label_to_names)

def decode_from_tfrecord(filenames):
	filename_queue = tf.train.string_input_producer(filenames)

	reader = tf.TFRecordReader()
	_, value = reader.read(filename_queue)
	features = tf.parse_single_example(
		value,
       features={
           'image/encoded': tf.FixedLenFeature([], tf.string),
           'image/classes/label': tf.FixedLenFeature([1],tf.int64),
	       'image/format': tf.FixedLenFeature([], tf.string)})
	image = tf.image.decode_jpeg(features['image/encoded'])
	return image


def _preprocessing_image (image):
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(name='resnet', is_training=True)
    image = image_preprocessing_fn(image, _IMAGE_SIZE, _IMAGE_SIZE)
    return image


def get_images_data ():
    data_set = _get_split(_SPLITE_NAME,
                          dataset_dir=_DATASET_DIR,
                          file_pattern=_FILE_PATTERN,
                          reader=None,
                          split_to_size=_SPLITS_TO_SIZES,
                          items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
                          num_classes=_NUM_CLASSES)
    provider = slim.dataset_data_provider.DatasetDataProvider(
        data_set,
        num_readers=4,
        common_queue_capacity=20 * _BATCH_SIZE,
        common_queue_min=10 * _BATCH_SIZE)
    [image, label] = provider.get(['image', 'label'])
    image = _preprocessing_image(image)
    images, labels = tf.train.batch(
        [image, label],
        batch_size=_BATCH_SIZE,
        num_threads=4,
        capacity=5 * _BATCH_SIZE)
    labels = tf.squeeze(labels, 1)
    labels = slim.one_hot_encoding(labels, _NUM_CLASSES)
    batch_queue = slim.prefetch_queue.prefetch_queue(
        [images, labels], capacity=2)
    image_batch, label_batch = batch_queue.dequeue()

    return image_batch, label_batch


image_batch, label_batch = get_images_data()
# filename = os.path.join(_DATASET_DIR, 'CatVSDog_test.tfrecord')
# image = decode_from_tfrecord([filename])
if __name__ == '__main__':

    with tf.Session() as sess:
        # sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        with tf.device('gpu:0'):
            for i in range(10):
                image_, label_ = sess.run([image_batch, label_batch])
                print(label_.shape)
                print(label_)
        coord.request_stop()
        coord.join()
