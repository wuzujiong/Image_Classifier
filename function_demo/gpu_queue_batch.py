import tensorflow as tf
from tensorflow.contrib import slim

from train_classifier.train_config import train_config
from datasets import dataset_factory
from preprocessing_image import preprocessing_factory

def get_batch_data():
    dataset = dataset_factory.get_dataset(
                train_config['dataset_name'],
                train_config['dataset_split_name'],
                train_config['dataset_dir'])

    provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=train_config['num_readers'],
                common_queue_capacity=20 * train_config['batch_size'],
                common_queue_min=10 * train_config['batch_size'])

    image_preprocessing_name = train_config['preprocessing_name'] or train_config['model_name']
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        image_preprocessing_name, is_training=True)


    [image, label] = provider.get(['image', 'label'])
    label -= train_config['labels_offset']
    train_image_size = train_config['train_image_size']
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
    image_batch, label_batch = batch_queue.dequeue()
    return image_batch, label_batch

image_batch, label_batch = get_batch_data()

if __name__ == '__main__':
    with tf.Session() as sess:
        tf.train.start_queue_runners()
        with tf.device('gpu:0'):
            for i in range(10):
                images_, labels_ = sess.run([image_batch, label_batch])
                print(images_.shape)