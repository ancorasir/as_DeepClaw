
"""Train the google's grasp network.

This scrip uses google's TFRecords files containing tf.train.Example protocol buffers.
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_train.py

By Fang Wan
"""
import tf_utils
import time
import os
import googleGrasp as gg
import tensorflow as tf

batch_size = 10
num_epochs = 2
learning_rate = 0.005
use_gpu_fraction = 0.8

checkpoint_path = './checkpoint'
summary_path = './summary'
data_path = '../tfrecord-3k'

def run_training():
    """Train googleGrasp"""
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # list of all the tfrecord files under /grasping_dataset_058/
        TRAIN_FILES = tf.train.match_filenames_once(os.path.join(data_path, '*.tfrecord'))
        # Input images and labels.
        images_batch, motions_batch, labels_batch = tf_utils.inputs(TRAIN_FILES, batch_size=batch_size, num_epochs=num_epochs)
        # Build a Graph that computes predictions from the inference model.
        is_training = tf.placeholder(tf.bool, name='is_training')
        logits = gg.inference(images_batch, motions_batch, is_training)
        # Add to the Graph the loss calculation.
        loss = gg.loss(logits,labels_batch)

        # Add GPU config, now maximun using 80% GPU memory to train
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = use_gpu_fraction

        # Add to the Graph operations that train the model.
        train_op = gg.training(loss, learning_rate)
        
        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session(config=config)
        sess.run(init_op)

        # Add summary writer
        tf.summary.scalar('loss', loss)
        merged_summary_op = tf.summary.merge_all()
        if not os.path.isdir(summary_path):
            os.mkdir(summary_path)
        summary_writer = tf.summary.FileWriter(summary_path, graph=tf.get_default_graph())

        # Add saver
        saver = tf.train.Saver()

        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)

        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint)

        # Start input enqueue threads.
        # Queue runner is a thread that uses a session and calls an enqueue op over and over again.
        # start_queue_runners starts threads for all queue runners collected in the graph
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()
                # Run one step of the model
                _, loss_value, summary = sess.run([train_op, loss, merged_summary_op], feed_dict={is_training: True})
                # Use TensorBoard to record 
                summary_writer.add_summary(summary)
                duration = time.time() - start_time
                if step % 10 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
        # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        saver.save(sess, checkpoint_path + '/Network', global_step=step)
        sess.close()

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()

