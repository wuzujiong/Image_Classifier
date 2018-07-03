"""
Train network util
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim

from datasets import data_batch


def congiure_learning_rate(num_samplees_per_epoch, batch_size, global_step):
    decay_steps = int(num_samplees_per_epoch / batch_size)