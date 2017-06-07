
"""Evaluate grasp network.

By Fang Wan
"""
import tf_utils
import time
import os
from graspNet import model as grasp_net
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# 40, 10min/epoch;
batch_size = 500
num_epochs = 1
use_gpu_fraction = 1

checkpoint_path = './checkpoint'
data_path = '/home/ancora-sirlab/wanfang/cropped_image'

def evaluate():
    with tf.Graph().as_default():
        TEST_FILES = tf.train.match_filenames_once(os.path.join(data_path, '*.tfrecord'))
	# Input images and labels.
        images_batch, indicators_batch, labels_batch = tf_utils.inputs(TEST_FILES, batch_size=batch_size, num_epochs=num_epochs, is_train=0)
        # Build a Graph that computes predictions from the inference model.
        model = grasp_net()
        model.initial_weights(weight_file='./bvlc_alexnet.npy')
        logits = model.inference(images_batch, indicators_batch)
        y = tf.nn.softmax(logits)

	# accuracy of the trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels_batch, 1))
	total_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Add GPU config, now maximun using 80% GPU memory to train
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
        step = 0
        Y_value = np.array([[0,1]])
        Label_value = np.array([[0,1]])
        try:
            while not coord.should_stop():
                y_value, labels_value, accuracy_value = sess.run([y, labels_batch, total_accuracy])
                Y_value = np.concatenate([Y_value, y_value])
                Label_value = np.concatenate([Label_value, labels_value])
                print('Total accuracy of %sth batch: %.3f' %(step+1,accuracy_value))
		step += 1
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
            np.savez('evaluation_prediction', Y_value=Y_value, Label_value=Label_value)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

def main(_):
    evaluate()

if __name__ == '__main__':
    tf.app.run()

