''' Train the image classifier '''
import tensorflow as tf
from tensorflow.contrib import slim
from deployment import model_deploy
from train_classifier.train_config import train_config
from datasets import dataset_factory
from networks import nets_factory
from preprocessing_image import preprocessing_factory



with tf.Graph().as_default():

	depploy_config = model_deploy.DeploymentConfig(
		num_clones=1,
		clone_on_cpu=False,
		replica_id=0,
		num_replicas=1,
	)
	sess = tf.Session()
	losses = tf.constant(2.2, dtype=tf.float32)
	logits = tf.constant(777, dtype=tf.float32)
	# tf.add_to_collection(tf.GraphKeys.SUMMARIES, logits)
	# tf.add_to_collection(tf.GraphKeys.SUMMARIES, losses)
	summeries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

	summeries.add(tf.summary.scalar('learning_rate', losses))

	# print('summary fo collection: ', sess.run(summeries))

	print(tf.GraphKeys)
	print('optimizer device: ', depploy_config.optimizer_device())
	print('variable_device: ', depploy_config.variables_device())
	print('clone_device: ', depploy_config.clone_device(0))

