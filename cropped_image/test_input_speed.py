import tf_utils
import time
import os
import tensorflow as tf
import datetime

batch_size = 100
num_epochs = 5

data_path = '/home/ancora-sirlab/wanfang/cropped_image/croppedImage_tfrecord'

TRAIN_FILES = tf.train.match_filenames_once(os.path.join(data_path, '*.tfrecord'))
# Input images and labels.
images_batch, indicators_batch, labels_batch = tf_utils.inputs(TRAIN_FILES, batch_size=batch_size, num_epochs=num_epochs)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess = tf.Session()
sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(20):
    #img = sess.run(images_batch)
    num1 = sess.run("input/shuffle_batch/random_shuffle_queue_Size:0")
    start_time = datetime.datetime.now()
    time.sleep(1)
    end_time = datetime.datetime.now()
    sec = (end_time - start_time).total_seconds()
    num2 = sess.run("input/shuffle_batch/random_shuffle_queue_Size:0")
    speed = (num2-num1)/sec
    print('Input speed is %s/s'%speed)

coord.request_stop()
coord.join(threads)
sess.close()

