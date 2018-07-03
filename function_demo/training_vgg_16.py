
''' Train the image classifier '''
import tensorflow as tf
from tensorflow.contrib import slim

from model_deployment import model_deploy
from train_classifier.train_config import train_config
from datasets import dataset_factory
from networks import nets_factory
from preprocessing_image import preprocessing_factory
from datasets import data_batch

def configure_learning_rate(num_samples_per_epoch, global_step):
	''' Configures the learning rate
	Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
	'''
	decay_steps = int(num_samples_per_epoch / train_config['batch_size'] *
	                  train_config['num_epochs_per_decay'])
	if train_config['sync_replicas']:
		decay_steps /= train_config['replicas_to_aggregate']

	if train_config['learning_rate_decay_type'] == 'exponential':
		return tf.train.exponential_decay(train_config['learning_rate'],
		                                  global_step,
										  decay_steps,
		                                  train_config['learning_rate_decay_factor'],
		                                  staircase=True,
		                                  name='exponential_decay_learning_rate')
	elif train_config['learning_rate_decay_type'] == 'fixed':
		return tf.constant(train_config['learning_rate'])
	elif train_config['learning_rate_decay_type'] == 'polynomial':
		return tf.train.polynomial_decay(train_config['learning_rate'],
		                                 global_step,
		                                 decay_steps,
		                                 train_config['end_learning_rate'],
		                                 power=1.0,
		                                 cycle=False,
		                                 name='polynomial_decay_learning_rate')
	else:
		raise ValueError('learning_rate_decay_type [%s] was not recognized' % train_config['learning_rate_decay_type'])

def configure_optimizer(learning_rate):
	"""Configures the optimizer used for training.

	 Args:
	   learning_rate: A scalar or `Tensor` learning rate.

	 Returns:
	   An instance of an optimizer.

	 Raises:
	   ValueError: if FLAGS.optimizer is not recognized.
	 """
	if train_config['optimizer'] == 'adadelta':
		optimizer = tf.train.AdadeltaOptimizer(learning_rate,
		                                       rho=train_config['adadelta_rho'],
		                                       epsilon=train_config['opt_epsilon'])
	elif train_config['optimizer'] == 'dadgrad':
		optimizer = tf.train.AdagradDAOptimizer(
			learning_rate,
			initial_gradient_squared_accumulator_value=train_config['adagrad_initial_accumulator_value'])
	elif train_config['optimizer'] == 'adam':
		optimizer = tf.train.AdamOptimizer(
			learning_rate,
			beta1=train_config['adam_beta1'],
			beta2=train_config['adam_beta2'],
			epsilon=train_config['opt_epsilon'])
	elif train_config['optimizer'] == 'ftrl':
		optimizer = tf.train.FtrlOptimizer(
			learning_rate,
			learning_rate_power=train_config['ftrl_learning_rate_power'],
			initial_accumulator_value=train_config['ftrl_initial_accumulator_value'],
			l1_regularization_strength=train_config['ftrl_l1'],
			l2_regularization_strength=train_config['ftrl_l2'])
	elif train_config['optimizer'] == 'momentum':
		optimizer = tf.train.MomentumOptimizer(
			learning_rate,
			momentum=train_config['momentum'],
			name='Momentum')
	elif train_config['optimizer'] == 'rmsprop':
		optimizer = tf.train.RMSPropOptimizer(
			learning_rate,
			decay=train_config['rmsprop_decay'],
			momentum=train_config['rmsprop_momentum'],
			epsilon=train_config['opt_epsilon'])
	elif train_config['optimizer'] == 'sgd':
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	else:
		raise ValueError('Optimizer [%s] was not recognized' % train_config['optimizer'])
	return optimizer

def get_variable_to_train():
	''' Return a list of variable to train
	Returns:
		A list of variable to train by the optimizer
	'''
	if train_config['trainable_scopes'] is None:
		return tf.trainable_variables()
	else:
		scopes = [scope.strip() for scope in train_config['trainable_scopes'].split(',')]
	variable_to_train = []
	for scope in scopes:
		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
		variable_to_train.append(variables)
	return variable_to_train

def get_init_fn():
	"""Returns a function run by the chief worker to warm-start the training.

	 Note that the init_fn is only run when initializing the model during the very
	 first global step.

	 Returns:
	   An init function run by the supervisor.
	 """
	if train_config['checkpoint_path'] is None:
		return None
	# Warn the user if a checkpoint exists in the train_dir. Then we'll be
	# ignoring the checkpoint anyway.
	if tf.train.latest_checkpoint(train_config['checkpoint_path']):
		tf.logging.info(
			'Ignoring --checkpoint_path because a checkpoint already exists in %s'
			% train_config['checkpoint_path'])
		return None

	exclusions = []
	if train_config['checkpoint_exclude_scopes']:
		exclusions = [scope.strip()
		              for scope in train_config['checkpoint_exclude_scopes'].split(',')]

		variable_to_restore = []
		for var in slim.get_model_variables():
			excluded = False
			for exclusion in exclusions:
				if var.op.name.startswith(exclusion):
					excluded = True
					break
			if not excluded:
				variable_to_restore.append(var)

		if tf.gfile.IsDirectory(train_config['checkpoint_path']):
			checkpoint_path = tf.train.latest_checkpoint(train_config['checkpoint_path'])
		else:
			checkpoint_path = train_config['checkpoint_path']

		tf.logging.info('Fune-tuning from %s' % checkpoint_path)

		return slim.assign_from_checkpoint_fn(
			checkpoint_path,
			variable_to_restore,
			ignore_missing_vars=train_config['ignore_missing_vars'])

def train():

	with tf.Graph().as_default():
		if not train_config['dataset_dir']:
			raise ValueError('You must set the dataset directory in "train_config.py"')
		tf.logging.set_verbosity(tf.logging.INFO)

		# Create global steps
		global_step = tf.train.create_global_step()
		image_batch, label_batch, dataset = data_batch.get_batch_data()
		# Select the network model
		network_fn = nets_factory.get_network_fn(
			train_config['model_name'],
			num_classes=(dataset.num_classes - train_config['labels_offset']),
			weight_decay=train_config['weight_decay'],
			is_training=True)

		logits, end_points = network_fn(image_batch)
		slim.losses.softmax_cross_entropy(logits, label_batch,
												   label_smoothing=train_config['label_smoothing'],
												   weights=1.0)
		total_loss = slim.losses.get_total_loss()
		tf.summary.scalar('lossed/total_loss', total_loss)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_tensor = slim.learning.create_train_op(total_loss, optimizer)
		slim.learning.train(
			train_tensor,
			optimizer)


if __name__ == '__main__':
	train()




