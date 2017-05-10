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


def tf_writer(src_folder, dis_folder, crop_box):
    """
    change the datas under src_folder into TFRecord fomat

    Args:
        src_folder: source folder with image and csv datas
        dis_folder: directory to save tfrecord file
        crop_box: cropping area for tray, box=(left, up, right, down)
    """
    if not os.path.isdir(dis_folder):
        print('dis folder not exist! now creating a new one ...')
        os.mkdir(dis_folder)
    writer = tf.python_io.TFRecordWriter(os.path.join(dis_folder, src_folder.split('/')[-1] + '.tfrecord'))

    data = pd.read_csv(os.path.join(src_folder, 'data.csv'))
    
    for i in range(0, 40):
        img_base = os.path.join(src_folder, 'I_'+str(i+1))
        if not os.path.isfile(img_base+'_1_color_camB.jpg'):
            print('file not exist!')
            break
        # crop the image before store
        #img_1 = np.array(Image.open(img_base+'_1_color_camB.jpg').crop(crop_box).resize((500, 500), Image.ANTIALIAS))
        img_01 = np.array(Image.open(img_base+'_01_color_camB.jpg').crop(crop_box).resize((500, 500), Image.ANTIALIAS))
        img_00 = np.array(Image.open(img_base+'_00_color_camB.jpg').crop(crop_box).resize((500, 500), Image.ANTIALIAS)).tobytes()
        #img_11 = np.array(Image.open(img_base+'_11_color_camB.jpg').crop(crop_box).resize((500, 500), Image.ANTIALIAS)).tobytes()
        #img_12 = np.array(Image.open(img_base+'_12_color_camB.jpg').crop(crop_box).resize((500, 500), Image.ANTIALIAS)).tobytes()
        #img_13 = np.array(Image.open(img_base+'_13_color_camB.jpg').crop(crop_box).resize((500, 500), Image.ANTIALIAS)).tobytes()
        # save image height and width
        height, width = img_01.shape[0], img_01.shape[1]
        img_01 = img_01.tobytes()
        # save depth array
        dp_file = np.load(img_base+'_1_depth_camB.npy')
        depth_height, depth_width = dp_file.shape[0], dp_file.shape[1]

        example = tf.train.Example(features=tf.train.Features(
            feature={
            'name': _bytes_feature(img_base),
            'img_01': _bytes_feature(img_01),
            #'img_1' : _bytes_feature(img_1),
            'img_00': _bytes_feature(img_00),
            #'img_11': _bytes_feature(img_11),
            #'img_12': _bytes_feature(img_12),
            #'img_13': _bytes_feature(img_13),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'move_00': _floats_feature(eval(data['move_00'][i][1:])),
            'move_1': _floats_feature(eval(data['move_1'][i][1:])),
            'rotate_angle': _floats_feature([data['rotate_angle'][i]]),
            #'gACT': _int64_feature(data['gACT'][i]),
            #'gMOD': _int64_feature(data['gMOD'][i]),
            #'gGTO': _int64_feature(data['gGTO'][i]),
            #'gSTA': _int64_feature(data['gSTA'][i]),
            #'gIMC': _int64_feature(data['gIMC'][i]),
            #'gFLT': _int64_feature(data['gFLT'][i]),
            #'gPRE': _int64_feature(data['gPRE'][i]),
            #'tcp_force': _floats_feature(eval(data['tcp_force'][i][1:])),
            'success': _floats_feature([data['success'][i]]),
            'depth_height': _int64_feature(depth_height),
            'depth_width': _int64_feature(depth_width),
            'depth_array_1': _bytes_feature(dp_file.tobytes()),
            'depth_array_01': _bytes_feature(np.load(img_base+'_01_depth_camB.npy').tobytes()),
            'depth_array_00': _bytes_feature(np.load(img_base+'_00_depth_camB.npy').tobytes()),
            #'depth_array_11': _bytes_feature(np.load(img_base+'_11_depth_camB.npy').tobytes()),
            #'depth_array_12': _bytes_feature(np.load(img_base+'_12_depth_camB.npy').tobytes()),
            #'depth_array_13': _bytes_feature(np.load(img_base+'_13_depth_camB.npy').tobytes())
            }))
        writer.write(example.SerializeToString())

    writer.close()
    return

def tf_reader(tf_record_filename_queue):
    """
    read all TFRecord files in the src_folder
    """

    tf_record_reader = tf.TFRecordReader()
    _, tf_record_serialized = tf_record_reader.read(tf_record_filename_queue)

    tf_record_features = tf.parse_single_example(
        tf_record_serialized,
        features={
        'move_00': tf.FixedLenFeature([6], tf.float32),
        'move_1': tf.FixedLenFeature([6], tf.float32),
        'success': tf.FixedLenFeature([], tf.float32),
        'img_01': tf.FixedLenFeature([], tf.string),
        'img_00': tf.FixedLenFeature([], tf.string),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'rotate_angle': tf.FixedLenFeature([1], tf.float32)
        })

    height = tf.cast(tf_record_features['height'], tf.int32)
    width = tf.cast(tf_record_features['width'], tf.int32)
    # TODO: use img_00 for grasp in the newest raw data
    grasp = tf.decode_raw(tf_record_features['img_00'], tf.uint8)
    grasp_0 = tf.decode_raw(tf_record_features['img_01'], tf.uint8)
    #grasp_1 = tf.decode_raw(tf_record_features['img_01'], tf.uint8)

    img_shape = tf.stack([height, width, 3])

    grasp = tf.reshape(grasp, img_shape)
    grasp_0 = tf.reshape(grasp_0, img_shape)
    #grasp_1 = tf.reshape(grasp_1, img_shape)
    # not use random crop 
    cropped_grasp = tf.random_crop(grasp, [472, 472, 3])
    cropped_grasp_0 = tf.random_crop(grasp_0, [472, 472, 3])
    #cropped_grasp_1 = tf.random_crop(grasp_1, [472, 472, 3])
    # concate the images
    images = tf.stack(
                      [tf.concat([cropped_grasp, cropped_grasp_0], 0)
                       ], axis=0
                      )
    images = tf.cast(images, tf.float32) # [1,472*2,472,3]
    # calculate input motions to the network
    motion0 = tf_record_features['move_1'] - tf_record_features['move_00']
    # append rotate angle to the motion list
    motion0 = motion0[:2]
    motion0 = tf.concat([motion0, tf_record_features['rotate_angle']], 0)
    motions = tf.stack([motion0], axis=0)
    # duplicate labels to the same size as images input
    label = tf_record_features['success']

    labels = tf.tile(tf.reshape(label, [1, 1]),
                     [images.get_shape()[0].value, 1]
                     ) #[2,1]
    labels = tf.cast(labels, tf.float32)
    return images, motions, labels


def inputs(filenames, batch_size, num_epochs):
    """Reads input data num_epochs times.

    Args:
        filename: a list of all file names which are used for training.
        batch_size: Number of examples per batch.
        num_epochs: Number of times to read the input data, or 0/None to train forever.

    Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3].
        * labels is a float tensor with shape [batch_size] with the true label.
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



