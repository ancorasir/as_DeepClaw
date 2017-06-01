import tensorflow as tf 
import numpy as np 
from PIL import Image
import pandas as pd
import os

# src_folder = ''


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tf_reader(tf_record_filename_queue):
    """
    read all TFRecord files in the src_folder
    """

    tf_record_reader = tf.TFRecordReader()
    _, tf_record_serialized = tf_record_reader.read(tf_record_filename_queue)

    tf_record_features = tf.parse_single_example(
        tf_record_serialized,
        features={
        'success': tf.FixedLenFeature([], tf.float32),
        'img_1': tf.FixedLenFeature([], tf.string),
        'img_00': tf.FixedLenFeature([], tf.string),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'rotate_angle': tf.FixedLenFeature([1], tf.float32)
        })

    height = tf.cast(tf_record_features['height'], tf.int32)
    width = tf.cast(tf_record_features['width'], tf.int32)
    # TODO: use img_00 for grasp in the newest raw data
    grasp = tf.decode_raw(tf_record_features['img_00'], tf.uint8)
    grasp_0 = tf.decode_raw(tf_record_features['img_1'], tf.uint8)

    img_shape = tf.stack([height, width, 3])

    grasp = tf.reshape(grasp, img_shape)
    grasp_0 = tf.reshape(grasp_0, img_shape)
    image = tf.stack(
                      [tf.concat([grasp, grasp_0], 0)
                       ], axis=0
                      )

    # duplicate labels to the same size as images input
    label = tf_record_features['success']
    label = tf.cast(label, tf.float32)

    theta = tf.cast(tf_record_features['rotate_angle'], tf.float32)
    return image, theta, label


def inputs(filenames, batch_size, num_epochs):
    """Reads input data num_epochs times.

    Args:
        filename: a list of all file names which are used for training.
        batch_size: Number of examples per batch.
        num_epochs: Number of times to read the input data, or 0/None to train forever.

    Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3].
        * labels is a float tensor with shape [batch_size, 1] with the true label.
        Note that an tf.train.QueueRunner is added to the graph, which
        must be run using e.g. tf.train.start_queue_runners().

    By Fang Wan
    """
    if not num_epochs: num_epochs = None
    with tf.name_scope('input'):
        # creates a FIFO queue for holding the filenames until the reader needs them. 
        # string_input_producer has options for shuffling and setting a maximum number of epochs. 
        # A queue runner adds the whole list of filenames to the queue once for each epoch. 
        # We grab a filename off our queue of filenames and use it to get examples from a TFRecordReader. 
        # Both the queue and the TFRecordReader have some state to keep track of where they are.
        # On initialization filename queue is empty. This is where the concept of QueueRunners comes in.
        # It is simply a thread that uses a session and calls an enqueue op over and over again.
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        images, motions, labels = tf_reader(filename_queue)
        # groups examples into batches randomly
        # shuffle_batch constructs a RandomShuffleQueue and proceeds to fill it with individual image and labels. 
        # This filling is done on a separate thread with a QueueRunner.
        # The RandomShuffleQueue accumulates examples sequentially until it contains batch_size +min_after_dequeue examples are present.
        # It then selects batch_size random elements from the queue to return.
        images_batch, motions_batch, labels_batch = tf.train.shuffle_batch([images, motions, labels], batch_size=batch_size, capacity=50 + 3 * batch_size, min_after_dequeue=50)
        actual_batch_size = images_batch.get_shape()[0].value * images_batch.get_shape()[1].value
        return tf.reshape(images_batch, [actual_batch_size, 472*2, 472, 3]), tf.reshape(motions_batch, [actual_batch_size, 3]), tf.reshape(labels_batch, [actual_batch_size, 1])



