import tensorflow as tf
import numpy as np
import os

image_name = '/Users/prmeasure/Desktop/Dogs vs. Cats Redux/images/dog.10362.jpg'

def _process_image(filename, sess):
	image_data = tf.gfile.FastGFile(filename, 'rb').read()
	image = tf.image.decode_jpeg(image_data)
	shape = sess.run(image).shape

	dir, image_name = os.path.split(filename)
	start_index = len(dir) + 1
	if filename[start_index:start_index + 3] == 'cat':
		label = int(0)
	elif filename[start_index:start_index + 3] == 'dog':
		label = int(1)
	else:
		raise ValueError('Label Error for classes')

	return image_data, shape, label

sess = tf.Session()
image_data, shape, label = _process_image(image_name, sess)

print(sess.run(tf.shape(image_data)))
print(shape)
print(label)