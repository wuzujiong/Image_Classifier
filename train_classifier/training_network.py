''' Train the image classifier '''
import tensorflow as tf
from tensorflow.contrib import slim
from train_classifier.train_config import train_config

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
		              for scope in train_config['checkpoint_exclude_scopes'].split()]