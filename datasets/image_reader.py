import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as datetime
from scipy.misc import imread, imresize

data_dir = 'train'
TFRecoder_filename = 'TFRecorder_CatVSDog_Train_Data'
training_image_name = 'CatVSDog_train.tfrecord'
testing_image_name = 'CatVSDog_test.tfrecord'
_NUM_TRAIN = 20000
_NUM_TEST = 5000

def get_filenames(data_dir):
	if not data_dir:
		raise ValueError('Fail to find filepath: ' + data_dir)
	filenames = os.listdir(data_dir)
	image_names = []
	for filename in filenames:
		_, extension = os.path.splitext(filename)
		if extension == '.jpg':
			image_name = os.path.join(data_dir, filename)
			if not tf.gfile.Exists(image_name):
				raise ValueError('Fail to find the file: ' + image_name)
			image_names.append(image_name)
	return image_names



def cover_image_to_TFRecorder(TFRecoder_filename, image_names, height, width):
	class image_recorder(object):
		'''create a class to save image info'''
		pass
	dir, image_name = os.path.split(image_names[0])
	start_index = len(dir) + 1
	all_counts = len(image_names)
	writer = tf.python_io.TFRecordWriter(TFRecoder_filename)

	for index, image_name in enumerate(image_names):
		img = imread(image_name)
		img = imresize(img, (height, width))

		if image_name[start_index:start_index + 3] == 'cat':
			label = int(0)
		elif image_name[start_index:start_index + 3] == 'dog':
			label = int(1)
		else:
			raise ValueError('Label Error for class')
		image = img.tostring()

		example = tf.train.Example(features=tf.train.Features(feature={
			'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
			'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
		}))
		writer.write(example.SerializeToString())
		if index % 100 == 0:
			print('Processing progress: {}/{}'.format(index, all_counts))
	writer.close()

def read_and_decode_image(filenames, image_shape):
	class image_record(object):
		pass
	result = image_record()
	filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
	reader = tf.TFRecordReader()
	key, value = reader.read(filename_queue)
	features = tf.parse_single_example(value, features={
		'image': tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.int64)
	})

	image = tf.decode_raw(features['image'], tf.uint8)
	image = tf.reshape(image, image_shape)
	# image = tf.cast(image, tf.float32)
	label = tf.cast(features['label'], tf.int32)
	result.label = label
	result.unit8image = image
	return result

def _gernerate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
	# create a queue that the shuffle the example, and then read 'batch_size'
	# images + labels from the example queue.

	num_preprocess_threads = 16
	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size,
			min_after_dequeue=min_queue_examples)
	else:
		images, label_batch = tf.train.batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size)
	tf.summary.image('image', images)

	return images, tf.reshape(label_batch, [batch_size])



height = 374
width = 500
image_shape = [height, width, 3]
image_names = get_filenames(data_dir)
cover_image_to_TFRecorder(TFRecoder_filename, image_names, height, width)
result = read_and_decode_image([TFRecoder_filename], image_shape)

images, labels = _gernerate_image_and_label_batch(result.unit8image, result.label, 100, 8, shuffle=True)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	img, label = sess.run([images, labels])
	coord.request_stop()
	coord.join()
	plt.figure()
	for i in range(8):
		print(label[i])
		plt.imshow(img[i])
		plt.show()













