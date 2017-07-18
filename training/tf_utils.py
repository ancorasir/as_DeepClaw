import tensorflow as tf 
import numpy as np 
from PIL import Image
import pandas as pd
import os

NUM_THETAS = 18
NUM_CLASSES = 2
#IMAGE_MEAN = np.array([104., 117., 124.])
IMAGE_MEAN = 164.0 # true mean, but train accuray is below 0.9
SIZE = 20

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
        'img_00': tf.FixedLenFeature([], tf.string),
        'rotate_angle': tf.FixedLenFeature([1], tf.float32)
        })
    images = []
    for i in range(size):
    #for i in tf.unstack(tf_record_features['img_00']):
        img = tf_record_features['img_00'][i]
        grasp = tf.decode_raw(img, tf.uint8)
        image = tf.reshape(grasp, [227, 227,3])
        images.append(image)
    images = tf.cast(tf.reshape(images,[size, 227, 227, 3]), tf.float32)
    images -= IMAGE_MEAN
    labels = tf.cast(tf_record_features['success'], tf.float32)  #[size]
    thetas = tf.cast(tf_record_features['rotate_angle'], tf.float32) + 3.14 #[size]
    return images, tf.reshape(thetas,[-1,1]), tf.reshape(labels,[-1,1])

def tf_reader_1(tf_record_filename_queue, size):
    """
    read all TFRecord files in the src_folder for traversed data
    """
    tf_record_reader = tf.TFRecordReader()
    _, tf_record_serialized = tf_record_reader.read_up_to(tf_record_filename_queue, size)
    tf_record_features = tf.parse_example(
        tf_record_serialized,
        features={
        'label': tf.FixedLenFeature([19], tf.float32),
        'img_00': tf.FixedLenFeature([], tf.string),
        })
    images = []
    for i in range(size):
        img = tf_record_features['img_00'][i]
        grasp = tf.decode_raw(img, tf.uint8)
        image = tf.reshape(grasp, [227, 227,3])
        images.append(image)
    images = tf.cast(tf.reshape(images,[size, 227, 227, 3]), tf.float32)
    images -= IMAGE_MEAN
    labels = tf.cast(tf_record_features['label'], tf.float32) #[size,NUM_THETAS+1]
    return images, labels

def inputs(filenames, batch_size, num_epochs, is_train=1):
    """Reads input data num_epochs times.

    Args:
        filename: a list of all file names which are used for training.
        batch_size: Number of examples per batch.
        num_epochs: Number of times to read the input data, or 0/None to train forever.
    """
    if not num_epochs: num_epochs = None
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        images, thetas, labels = tf_reader(filename_queue, size=SIZE)
	if is_train == 1:
            I, T, L = tf.train.shuffle_batch([images, thetas, labels], batch_size=batch_size, capacity=5000 + 10 * batch_size, min_after_dequeue=5000, num_threads=2, enqueue_many=True)
        if is_train == 0:
            I, T, L = tf.train.batch([images, thetas, labels], batch_size=batch_size, capacity=5000 + 3 * batch_size,  enqueue_many=True)

        T_index = tf.floordiv(T, 2*3.14/NUM_THETAS)+1

        # generate new data by traverse all thetas while predict on test set
        if 0:
            I_traversed = tf.tile(I, [NUM_THETAS, 1, 1, 1])
            indicators = tf.reshape(
                tf.tile(
                    tf.reshape(tf.range(1,NUM_THETAS+1),[NUM_THETAS,1]),
                    [1,batch_size] ),
                [-1,1])
            L_traversed = tf.reshape(tf.tile(L,[NUM_THETAS, 1]),[-1])
            return I_traversed, tf.to_float(indicators)*1000, tf.one_hot(tf.to_int32(L_traversed), NUM_CLASSES)

        return I, T_index*100, tf.one_hot(tf.to_int32(tf.reshape(L,[-1])), NUM_CLASSES)

