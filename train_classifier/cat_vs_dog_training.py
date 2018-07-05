
''' Train the image classifier '''
import tensorflow as tf
from tensorflow.contrib import slim

from train_classifier.train_config import train_config
from networks import nets_factory
from datasets import data_batch
import time
from datetime import datetime
import numpy as np
import os
from datasets import cat_vs_dog


_MAX_STEPS = 100000
_NUM_GPU = 1
_log_device_placement = False


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


def network_fn():
    network_fn = nets_factory.get_network_fn(
        name='resnet',
        num_classes=2,
        weight_decay=train_config['weight_decay'],
        is_training=True)
    return network_fn

def tower_loss(scope, image_batch, label_batch):
    model = network_fn()
    logits, endpoints = model(image_batch)
    # loss = slim.losses.softmax_cross_entropy(logits, label_batch)
    slim.losses.softmax_cross_entropy(
        logits,
        label_batch,
        label_smoothing=train_config['label_smoothing'],
        weights=1.0)
    losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)
    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.train.create_global_step()
        learning_rate = configure_learning_rate(20000, global_step)
		# learning_rate = 0.001
        optimizer = configure_optimizer(learning_rate)
        # Gather initial summaries
        # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # summaries.add(tf.summary.scalar('learning_rate', learning_rate))
        tower_grads = []
        for i in range(1):
            with tf.variable_scope(tf.get_variable_scope()):
                with tf.device('/gpu:%d'%i):
                    with tf.name_scope('resnet_loss%d'%i) as scope:
                        image_batch, label_batch = cat_vs_dog.get_images_data()
                        total_loss = tower_loss(scope, image_batch, label_batch)
                        tf.get_variable_scope().reuse_variables()
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = optimizer.compute_gradients(total_loss)
                        tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        summaries.append(tf.summary.scalar('learning_rate', learning_rate))
        for grad, var, in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        apply_gradient_op = optimizer.apply_gradients(grads, global_step)

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        variable_averages = tf.train.ExponentialMovingAverage(train_config['moving_average_decay'], global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op, variable_averages_op)

        saver = tf.train.Saver(tf.global_variables())
        summaries_op = tf.summary.merge(summaries)
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=True))

        sess.run(init)

        tf.train.start_queue_runners(sess=sess)
        summaries_writer = tf.summary.FileWriter(train_config['train_dir'], sess.graph)

        for step in range(_MAX_STEPS):
            start_time = time.time()
            _, loss_value = sess.run([train_op, total_loss])
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % 10 == 0:
                num_example_per_step = train_config['batch_size']
                example_per_sec = num_example_per_step / duration
                sec_per_batch = duration
                print('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f (sec/batch)' %
                      (datetime.now(), step, loss_value, example_per_sec, sec_per_batch))
            if step % 100 == 0:
                summary_str = sess.run(summaries_op)
                summaries_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or (step + 1) == _MAX_STEPS:
                checkpoint_path = os.path.join(train_config['train_dir'], 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

if __name__ == '__main__':
    if tf.gfile.Exists(train_config['train_dir']):
        tf.gfile.DeleteRecursively(train_config['train_dir'])
    tf.gfile.MakeDirs(train_config['train_dir'])
    train()





