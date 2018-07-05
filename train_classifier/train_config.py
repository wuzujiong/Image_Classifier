from tensorflow.contrib import slim
import tensorflow as tf

train_config = {
	'master': '', # The address of the TensorFlow master to use.
	'train_dir': 'F:/DL_Datasets/Dogs vs. Cats Redux/checkpoints', # Directory where checkpoints and event logs are writeten to.
	'num_clones': 1, # Number of model clones to deploy.
	'clone_on_cpu': False, # Use CPUs to deploy clones, False for GPU train.
	'worker_replicas': 1, # Number of worker replicas.
	'num_ps_tasks': 0, # The number of parameter servers.
	'num_readers': 4, # The number of parallel readers that read data from the dataset.
	'num_preprocessing_threads': 4, # The number of threads used to create the batches.
	'log_every_n_steps': 10, # The frequency with which logs are print.
	'save_summaries_secs': 600, # The frequency with which summaries are saved, in seconds
	'save_interval_secs': 600, # The frequency with which the model is saved, in seconds.
	'task': 0, # Task id of the replica running the training
	'weight_decay': 0.00004,
	'optimizer': 'rmsprop', #The name of the optimizer, one of "adadelta", "adagrad", "adam",'
                            #   '"ftrl", "momentum", "sgd" or "rmsprop".'
	'adadelta_rho': 0.95, # The decay rate for adadelta.
	'adagrad_initial_accumulator_value': 0.1, # Starting value for the AdaGrad accumulators.
	'adam_beta1': 0.9, # The exponential decay rate for the 1st moment estimates.
	'adam_beta2': 0.99, # The exponential decay rate for the 2nd moment estimates.
	'opt_epsilon': 1.0, # Epsilon term for the optimizer.
	'ftrl_learning_rate_power': -0.5, # The learning rate power.
	'ftrl_initial_accumulator_value': 0.1, # Starting value for the FTRL accumulators.
	'ftrl_l1': 0.0, # The FTRL l1 regularization strength.
	'ftrl_l2': 0.0, # The FTRL l2 regularization strength.
	'momentum': 0.9, # The momentum for the MomentumOptimizer and RMSPropOptimizer.
	'rmsprop_momentum': 0.9,
	'rmsprop_decay': 0.9, # Decay term for RMSProp.
	'learning_rate_decay_type': 'exponential',
	'learning_rate': 0.0001,  # Initial learning rate.
	'end_learning_rate': 0.00001,  # The minimal end learning rate used by a polynomial decay learning rate.
	'label_smoothing': 0.0,
	'learning_rate_decay_factor': 0.1,
	'num_epochs_per_decay': 2.0,  # Number of epochs after which learning rate decays.
	'sync_replicas': False,  # Whether or not to synchronize the replicas during training.
	'replicas_to_aggregate': 1,  # The Number of gradients to collect before updating params.
	'moving_average_decay': 0.9,  # The decay to use for the moving average.
	'dataset_name': 'cifar10', # The name of the dataset to load.
	'dataset_split_name': 'train', # The name of the train/test split.
	'dataset_dir': 'F:/DL_Datasets/cifar-10-python', # The directory where the dataset files are stored.
	'labels_offset': 0,  # 'An offset for the labels in the dataset. This flag is primarily used to '
                         # 'evaluate the VGG and ResNet architectures which do not use a background '
                         # 'class for the ImageNet dataset.'
	'model_name': 'resnet',
	'preprocessing_name': None, # If None, use the model name.
	'batch_size': 32,
	'train_image_size': 224,
	'max_number_of_steps': None,
	'checkpoint_path': None,  # The path to a checkpoint from which to fine-tune.
	'checkpoint_exclude_scopes': None,# 'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.
	'trainable_scopes': None,# Comma-separated list of scopes to filter the set of variables to train.'By default, None would train all the variables.
	'ignore_missing_vars': False,  # When restoring a checkpoint would ignore missing variables.
	'gpu_memory_fraction': 0.8, #GPU memory fraction to use.
}
