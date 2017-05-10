"""Evaluation for grasp network

Fang Wan
"""

import tf_utils
import googleGrasp_softmax as gg
import tensorflow as tf
import os
import numpy as np
import tensorflow.contrib.slim as slim

batch_size = 40
num_epochs = 1
checkpoint_path = './checkpoint_120epoch'
data_path = '/home/ancora-sirlab/wanfang/training_softmax/tfrecords-5k-test'

use_gpu_fraction = 0.9

def evaluate():
    with tf.Graph().as_default() as g:
        TRAIN_FILES = tf.train.match_filenames_once(os.path.join(data_path, '*.tfrecord'))
        # Input images and labels.
        images_batch, motions_batch, labels_batch = tf_utils.inputs(TRAIN_FILES, batch_size=batch_size, num_epochs=num_epochs)
        # Build a Graph that computes predictions from the inference model.
        is_training = tf.placeholder(tf.bool, name='is_training')
        logits = gg.inference(images_batch, motions_batch, is_training)
        y = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels_batch, 1))

        # Add GPU config, now maximun using 80% GPU memory to train
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = use_gpu_fraction
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session(config=config)
        sess.run(init_op)
        
	# Add saver
        variables = slim.get_variables_to_restore()
        saver = tf.train.Saver([v for v in variables if v.name != 'matching_filenames:0'])
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if checkpoint:
            saver.restore(sess, checkpoint)
        
	coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        total_sample_count = 0
	step = 0
        total_true_count = 0
        try:
            while not coord.should_stop():
		predictions = sess.run([correct_prediction], feed_dict={is_training: False})
                total_true_count += np.sum(predictions)
		total_sample_count += batch_size
		print(step)
                print('true count = %.3f' %np.sum(predictions))
		step += 1    
	except tf.errors.OutOfRangeError:
            print('Done test data')
        finally:
	    precision = total_true_count / float(total_sample_count)
            print('Total true_count =  %.3f' %total_true_count)
            print('Total sample count =  %.3f' %total_sample_count)
	    print('Precision @ 1 = %.3f' %precision)
        # When done, ask the threads to stop.
            coord.request_stop()
	coord.join(threads)

def main(argv=None):
    evaluate()

if __name__ == '__main__':
  tf.app.run()
