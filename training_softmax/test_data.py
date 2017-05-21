# read and check if training data is feeding to the trained network

import tf_utils
import time
import os
import googleGrasp_softmax as gg
import tensorflow as tf
import numpy as np

data_path = '/home/ancora-sirlab/wanfang/training_softmax/tfrecords-5k-train'
checkpoint_path = './checkpoint_120_backup'
TRAIN_FILES = tf.train.match_filenames_once(os.path.join(data_path, '*.tfrecord'))
batch_size = 30
num_epochs=1

images_batch, motions_batch, labels_batch = tf_utils.inputs(TRAIN_FILES, batch_size=batch_size, num_epochs=num_epochs)
is_training = tf.placeholder(tf.bool, name='is_training')
logits = gg.inference(images_batch, motions_batch, is_training)
y = tf.nn.softmax(logits)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess = tf.Session()
sess.run(init_op)
saver = tf.train.Saver()
checkpoint = tf.train.latest_checkpoint(checkpoint_path)
saver.restore(sess, checkpoint)

# compute and print accuracy of the sample data
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
images, motions, y_value,labels = sess.run([images_batch,motions_batch, y,labels_batch], feed_dict={is_training: False})
print('Train Accuracy = ', sum(np.rint(y_value[:,0])==labels[:,0])/30.0)
print('y_predict VS y_label:\n', np.stack([np.rint(y_value[:,0]),labels[:,0]],axis=1))
print(np.stack([y_value[:,0],labels[:,0]],axis=1))
np.mean(motions,axis=0)
np.var(motions,axis=0)

# save one train data sample for test in run_CEM
i = 2
img_00=images[i,:472,:,:]
img_01=images[i,472:,:,:]
motion=motions[i]
y_predict=y_value[i]
label=labels[i]
np.savez('outfile_fail023',img_00=img_00,img_01=img_01,motion= motion, y_predict=y_predict, label=label)
