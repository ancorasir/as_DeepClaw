import tensorflow as tf 
import numpy as np 
from PIL import Image
import pandas as pd
import os

NUM_THETAS = 18
NUM_CLASSES = 4

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tf_reader(tf_record_filename_queue, size):
    """
    read all TFRecord files in the src_folder
    """
    tf_record_reader = tf.TFRecordReader()
    _, tf_record_serialized = tf_record_reader.read_up_to(tf_record_filename_queue, size)
    tf_record_features = tf.parse_example(
        tf_record_serialized,
        features={
        'success': tf.FixedLenFeature([], tf.float32),
        'isDoll': tf.FixedLenFeature([], tf.float32),
        'img_00': tf.FixedLenFeature([], tf.string),
        'rotate_angle': tf.FixedLenFeature([1], tf.float32)
        })
    # TODO: use img_00 for grasp in the newest raw data
    images = []
    for i in range(size):
        img = tf_record_features['img_00'][i]
        grasp = tf.decode_raw(img, tf.uint8)
        image = tf.reshape(grasp, [360, 360,3])
        images.append(image)
    images = tf.cast(tf.reshape(images,[size, 360, 360, 3]), tf.float32)
    images = tf.image.resize_images(images, [227, 227])
    labels = tf.cast(tf_record_features['success'], tf.float32)
    isDolls = tf.cast(tf_record_features['isDoll'], tf.float32)
    thetas = tf.cast(tf_record_features['rotate_angle'], tf.float32)
    return images, thetas, tf.reshape(isDolls, [size,1]), tf.reshape(labels, [size,1])

def tf_reader_1(tf_record_filename_queue, size):
    """
    read all TFRecord files in the src_folder
    """
    tf_record_reader = tf.TFRecordReader()
    _, tf_record_serialized = tf_record_reader.read_up_to(tf_record_filename_queue, size)
    tf_record_features = tf.parse_example(
        tf_record_serialized,
        features={
        'label': tf.FixedLenFeature([19], tf.float32),
        'img_00': tf.FixedLenFeature([], tf.string),
        })
    # TODO: use img_00 for grasp in the newest raw data
    images = []
    for i in range(size):
        img = tf_record_features['img_00'][i]
        grasp = tf.decode_raw(img, tf.uint8)
        image = tf.reshape(grasp, [227, 227,3])
        images.append(image)
    images = tf.cast(tf.reshape(images,[size, 227, 227, 3]), tf.float32)
    labels = tf.cast(tf_record_features['label'], tf.float32) #[size,NUM_THETAS+1]
    return images, labels

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
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        images, labels = tf_reader_1(filename_queue, size=10)
        I, L = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=5000 + 3 * batch_size, min_after_dequeue=5000, enqueue_many=True)
        # generate new data by traverse all thetas
        I_traversed = tf.tile(I, [NUM_THETAS+1, 1, 1, 1])
        indicators = tf.reshape(
            tf.tile(
                tf.reshape(tf.range(NUM_THETAS+1),[NUM_THETAS+1,1]),
                [1,batch_size] ),
            [-1,1])
        L_traversed = tf.reshape(tf.transpose(L),[-1])
        return I_traversed, tf.to_float(indicators), tf.one_hot(tf.to_int32(L_traversed), NUM_CLASSES)

def traverse_theta(I, T, D, L, batch_size):
    """
       If use tf_reader to read the original tfrecords, use this function to generate traversed training sample bu call traverse_theta(I, T, D, L, batch_size) in funtion inputs

       Indicator {0:no object in the grasp area, 1:[0,20) degrees with object in the grasp area, 2:[20,40)}  
       Label {0:fail, 1:success, 2:unknown, 3:conflict}
    """
    images_batch = tf.tile(I, [NUM_THETAS+1, 1, 1, 1])
    indicators_batch = tf.reshape( 
        tf.tile( 
            tf.reshape(tf.range(NUM_THETAS+1),[NUM_THETAS+1,1]), 
            [1,batch_size] ), 
        [-1,1])
    true_indices = []
    for i in range(batch_size*(NUM_THETAS+1)):
        # image index
        ii = tf.mod(i, batch_size)
        indicator = tf.floordiv(i, batch_size)
        # find which grasp degree interval theta belongs to, theta in [-pi, pi)
        theta_idx = tf.floordiv(T[ii]+3.14, 3.14/NUM_THETAS) + 1

        hot = tf.case({
                tf.logical_and( tf.equal(D[ii][0],0.0), tf.equal(indicator,0) ): (lambda:tf.constant(0)), 
                tf.logical_and( tf.equal(D[ii][0],0.0), tf.greater(indicator,0) ): (lambda:tf.constant(3)),
                tf.logical_and( tf.equal(D[ii][0],1.0), tf.equal(indicator,0) ): (lambda:tf.constant(3)),
                # tf.logical_and( tf.equal(D[ii],1), tf.greater(indicator,0) ): (lambda:0,1,2)
                # indicator = 1,2,3,...,18; theta_idx = 1,2,3,...,18; L[ii] = 0,1 
                tf.equal(theta_idx[0], tf.to_float(indicator)): (lambda:tf.to_int32(L[ii][0])) 
                },
           default=(lambda:tf.constant(2)), exclusive=True)
        true_indices.append(hot)
    return images_batch,tf.to_float( indicators_batch), tf.one_hot(tf.stack(true_indices,0), NUM_CLASSES)
