''' Train the image classifier '''
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import slim

from model_deployment import model_deploy
from train_classifier.train_config import train_config
from datasets import dataset_factory
from networks import nets_factory
from preprocessing_image import preprocessing_factory
from networks.vgg_model import vgg_arg_scope

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
	if not train_config['dataset_dir']:
		raise ValueError('You must set the dataset directory in "train_config.py"')

	tf.logging.set_verbosity(tf.logging.INFO)

	with tf.Graph().as_default():
		#========================================================#
		# Config model deploy #
		#========================================================#
		# TODO: add a model deploy function and create global variable
		deploy_config = model_deploy.DeploymentConfig(
			num_clones=1,
			clone_on_cpu=False,
			replica_id=0,
			num_replicas=1,
			num_ps_tasks=0)

		# Create global variable
		print('deploy_config.variables_device():', deploy_config.variables_device())
		print('deploy_config.inputs_device(): ',deploy_config.inputs_device())
		print('deploy_config.clone_device(): ', deploy_config.clone_device(0))
		print('deploy_config.optimizer_device(): ', deploy_config.optimizer_device())
		print('deploy_config.caching_device(): ', deploy_config.caching_device())
		print(tf.get_default_graph)


		with tf.device(deploy_config.variables_device()):
			global_step = slim.create_global_step()
		#========================================================#
		# Select the dataset
		#=========================================================#
		dataset = dataset_factory.get_dataset(
			train_config['dataset_name'],
			train_config['dataset_split_name'],
			train_config['dataset_dir'])
		#========================================================#
		# Select the Network model
		#=========================================================#
		network_fn = nets_factory.get_network_fn(
			train_config['model_name'],
			num_classes=(dataset.num_classes - train_config['labels_offset']),
			weight_decay=train_config['weight_decay'],
			is_training=True)
		#========================================================#
		# Select the preprocessing function
		#=========================================================#
		preprocessing_name = train_config['preprocessing_name'] or train_config['model_name']
		# TODO implement the preprocessing func.
		image_preprocessing_name = train_config['preprocessing_name'] or train_config['model_name']
		image_preprocessing_fn = preprocessing_factory.get_preprocessing(
			image_preprocessing_name, is_training=True)

		#========================================================#
		# Create a dataset provider that loads data from the dataset
		#=========================================================#
		with tf.device(deploy_config.inputs_device()):
			with tf.name_scope(train_config['dataset_name'] + '_data_provider'):
				provider = slim.dataset_data_provider.DatasetDataProvider(
					dataset,
					num_readers=train_config['num_readers'],
					common_queue_capacity=20 * train_config['batch_size'],
					common_queue_min=10 * train_config['batch_size'])

			[image, label] = provider.get(['image', 'label'])
			label -= train_config['labels_offset']
			train_image_size = train_config['train_image_size'] or network_fn.default_image_size

			image = image_preprocessing_fn(image, train_image_size, train_image_size)

			images, labels = tf.train.batch(
				[image, label],
				batch_size=train_config['batch_size'],
				num_threads=train_config['num_preprocessing_threads'],
				capacity=5 * train_config['batch_size'])
			labels = slim.one_hot_encoding(
				labels,
				dataset.num_classes - train_config['labels_offset'])

			batch_queue = slim.prefetch_queue.prefetch_queue(
				[images, labels],
				capacity=2 * train_config['num_clones'])

		#========================================================#
		# Define the model
		#=========================================================#
		def clone_fn(batch_queue):
			images, labels = batch_queue.dequeue()

			# TODO: Add a model arg_scope for model.
			# arg_scope = vgg_arg_scope()
			# with slim.arg_scope(arg_scope):
			logits, end_points = network_fn(images)

		# ========================================================#
		# Define the model loss function
		# =========================================================#
		# 	if 'AuxLogits' in end_points:
		# 		slim.losses.softmax_cross_entropy(
		# 			end_points['AuxLogits'],
		# 			labels,
		# 			label_smoothing=train_config['label_smoothing'],
		# 			weights=0.4,
		# 			scope='aux_loss')
		# 	with tf.device('/device:CPU:0'):
		# 		slim.losses.softmax_cross_entropy(
		# 			logits,
		# 			labels,
		# 			label_smoothing=train_config['label_smoothing'],
		# 			weights=1.0)
			return end_points


		# Gather initial summaries
		summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

		clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
		first_clones_scope = deploy_config.clone_scope(0)

		# Gather update_ops from the first clone. These contain, for example,
		# the updates for the batch_norm variables created by network_fn.
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clones_scope)

		# Add summaries for end_points
		end_points = clones[0].outputs
		for end_point in end_points:
			x = end_points[end_point]
			summaries.add(tf.summary.histogram('activations/' + end_point, x))
			summaries.add(tf.summary.scalar('sparsity/' + end_point,
											tf.nn.zero_fraction(x)))

		# Add summaries for losses
		for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clones_scope):
			summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

		# Add summaries for variables.
		for variable in slim.get_model_variables():
			summaries.add(tf.summary.histogram(variable.op.name, variable))

		#========================================================#
		# Configure the moving averages
		#=========================================================#
		if train_config['moving_average_decay']:
			moving_average_variables = slim.get_model_variables()
			variable_averages = tf.train.ExponentialMovingAverage(
				train_config['moving_average_decay'],
				global_step)
		else:
			moving_average_variables, variable_averages = None, None
		#========================================================#
		# Configure optimization procedure
		#=========================================================#
		with tf.device(deploy_config.optimizer_device()):
			learning_rate = configure_learning_rate(dataset.num_samples, global_step)
			optimizer = configure_optimizer(learning_rate)
			summaries.add(tf.summary.scalar('learning_rate', learning_rate))

		if train_config['sync_replicas']:
			optimizer = tf.train.SyncReplicasOptimizer(
				opt=optimizer,
				replicas_to_aggregate=train_config['replicas_to_aggregate'],
				total_num_replicas=train_config['worker_replicas'],
				variable_averages=variable_averages,
				variables_to_average=moving_average_variables)
		elif train_config['moving_average_decay']:
			# Update ops executed locally by trainer
			update_ops.append(variable_averages.apply(moving_average_variables))

		# Variables to train.
		variable_to_train = get_variable_to_train()

		# Return a train_tensor and summary_op
		total_loss, clones_gradients = model_deploy.optimize_clones(
			clones,
			optimizer,
			var_list=variable_to_train)

		# Add total_loss to summary
		summaries.add(tf.summary.scalar('total_loss', total_loss))

		# Create gradient udates
		grad_updates = optimizer.apply_gradients(clones_gradients, global_step)

		update_ops.append(grad_updates)
		update_op = tf.group(*update_ops)

		with tf.control_dependencies([update_op]):
			train_tensor = tf.identity(total_loss, name='train_op')

		# Add the summaries from the first clone. These contain the summaries
		# created by model_fn and either optimize_clones() or _gather_clone_loss().
		summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
		                                   first_clones_scope))

		# Merge all summaries together.
		summary_op = tf.summary.merge(list(summaries), name='summary_op')

		###########################
		# Kicks off the training. #
		###########################
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=train_config['gpu_memory_fraction'])
		conig = tf.ConfigProto(log_device_placement=False,
							   gpu_options=gpu_options)
		saver = tf.train.Saver(max_to_keep=5,
							   keep_checkpoint_every_n_hours=0.1,
							   write_version=2,
							   pad_step_number=False)


		slim.learning.train(
			train_tensor,
			logdir=train_config['train_dir'],
			master='',
			is_chief=True,
			init_fn=get_init_fn(),
			summary_op=summary_op,
			number_of_steps=train_config['max_number_of_steps'],
			log_every_n_steps=train_config['log_every_n_steps'],
			save_summaries_secs=train_config['save_summaries_secs'],
			saver=saver,
			save_interval_secs=train_config['save_interval_secs'],
			# session_config=conig,
			sync_optimizer=None)

if __name__ == '__main__':
	train()




