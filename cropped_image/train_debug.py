import tf_utils
import time
import os
from graspNet import model as grasp_net
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# 40, 10min/epoch;
batch_size = 100
num_epochs = 1
use_gpu_fraction = 1

checkpoint_path = './checkpoint'
data_path = '/home/ancora-sirlab/wanfang/cropped_image'

TEST_FILES = tf.train.match_filenames_once(os.path.join(data_path, '*.tfrecord'))
images_batch, indicators_batch, labels_batch = tf_utils.inputs(TEST_FILES, batch_size=batch_size, num_epochs=num_epochs, is_train=0)
# Build a Graph that computes predictions from the inference model.
model = grasp_net()
model.initial_weights(weight_file='./bvlc_alexnet.npy')
logits = model.inference(images_batch, indicators_batch)
y = tf.nn.softmax(logits)
conv1 = model.conv1
conv2 = model.conv2
conv3 = model.conv3
conv4 = model.conv4
conv5 = model.conv5
fc1 = model.fc1
fc6 = model.fc6
fc7 = model.fc7

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = use_gpu_fraction
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess = tf.Session(config=config)
sess.run(init_op)

# Add saver
variables = slim.get_variables_to_restore()
saver = tf.train.Saver([v for v in variables if v.name != 'matching_filenames:0'])
checkpoint = tf.train.latest_checkpoint(checkpoint_path)
saver.restore(sess, checkpoint)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
yy, ii, ll,c1,c2,c3,c4, c5, f1 = sess.run([y, indicators_batch, labels_batch, conv1, conv2, conv3, conv4, conv5, fc1])

np.mean(f1)
hh=f1[:]
np.mean(hh[hh>0])
