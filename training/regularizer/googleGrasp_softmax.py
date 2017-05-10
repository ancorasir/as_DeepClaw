"""Builds the googleGrasp network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and apply gradients.

This file is used by "googleGrasp_train.py" and not meant to be run.
Reference:
/tensorflow/examples/tutorials/mnist.py
/tensorflow/tutorials/image/cifar10/cifar10.py

By Fang Wan
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm

# http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
# http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
# discussion https://github.com/tensorflow/tensorflow/issues/1122
# mnist_cnn_bn https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412
def BatchNorm(input, is_training=True, scope=None):
    bn_train = batch_norm(input, decay=0.999, center=True, scale=False,
                          updates_collections=None,
                          is_training=True,
                          reuse=None, # for training
                          trainable=True,
                          scope=scope)
    bn_inference = batch_norm(input, decay=0.999, center=True, scale=False,
                              updates_collections=None,
                              is_training=False,
                              reuse=True, # for testing
                              trainable=True,
                              scope=scope)
    z = tf.cond(is_training, lambda: bn_train, lambda: bn_inference)
    return z

# Construct model
def inference(images, motions, is_training,keep_prob):
    """Built the google grasp model up tp where it may be used for inference.

    Args:
        images: from inputs(), [batch, in_height, in_width, in_channels], in_height=472*2, width=472, in_channel=3
        motions: from inputs(), [batch, 5]
        is_training: bool placeholder, if inference is for training or testing

    Returns:
        logits
    """
    # conv1
    # weights's shape: [filter_height, filter_width, in_channels, out_channels]
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', [6, 6, 3, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(images, weights, strides = [1, 2, 2, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv1')
        conv1 = tf.nn.relu(conv_bn, name=scope.name)
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1],
                           padding='SAME', name='pool1')
    # 6 5*5 conv + Relu
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights', [5, 5, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(pool1, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv2')
        conv2 = tf.nn.relu(conv_bn, name=scope.name)
    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights', [5, 5, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(conv2, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv3')
        conv3 = tf.nn.relu(conv_bn, name=scope.name)
    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable('weights', [5, 5, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(conv3, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv4')
        conv4 = tf.nn.relu(conv_bn, name=scope.name)
    with tf.variable_scope('conv5') as scope:
        weights = tf.get_variable('weights', [5, 5, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(conv4, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv5')
        conv5 = tf.nn.relu(conv_bn, name=scope.name)
    with tf.variable_scope('conv6') as scope:
        weights = tf.get_variable('weights', [5, 5, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(conv5, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv6')
        conv6 = tf.nn.relu(conv_bn, name=scope.name)
    with tf.variable_scope('conv7') as scope:
        weights = tf.get_variable('weights', [5, 5, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(conv6, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv7')
        conv7 = tf.nn.relu(conv_bn, name=scope.name)
    pool2 = tf.nn.max_pool(conv7, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1],
                           padding='SAME', name='pool2') # [None,53,27,64]
    # fc1, processing motor command
    with tf.variable_scope('fc1'):
        # our motion vector is 7 dim
        weights = tf.get_variable('weights', [3, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(motions, weights) + biases)
        fc1 = tf.reshape(fc1,shape=[-1, 1, 1, 64])
    # pointwise addition of image input and motor command input
    x = tf.add(tf.tile(fc1, [1, 53, 27, 1]), pool2)
    # 6 3*3 conv layers
    with tf.variable_scope('conv8') as scope:
        weights = tf.get_variable('weights', [3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(x, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv8')
        conv8 = tf.nn.relu(conv_bn, name=scope.name)
    with tf.variable_scope('conv9') as scope:
        weights = tf.get_variable('weights', [3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(conv8, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv9')
        conv9 = tf.nn.relu(conv_bn, name=scope.name)
    with tf.variable_scope('conv10') as scope:
        weights = tf.get_variable('weights', [3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(conv9, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv10')
        conv10 = tf.nn.relu(conv_bn, name=scope.name)
    with tf.variable_scope('conv11') as scope:
        weights = tf.get_variable('weights', [3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(conv10, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv11')
        conv11 = tf.nn.relu(conv_bn, name=scope.name)
    with tf.variable_scope('conv12') as scope:
        weights = tf.get_variable('weights', [3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(conv11, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv12')
        conv12 = tf.nn.relu(conv_bn, name=scope.name)
    with tf.variable_scope('conv13') as scope:
        weights = tf.get_variable('weights', [3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(conv12, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv13')
        conv13 = tf.nn.relu(conv_bn, name=scope.name)
    pool3 = tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool3') # [None,27,14,64]
    # 3 3*3 conv layers
    with tf.variable_scope('conv14') as scope:
        weights = tf.get_variable('weights', [3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(pool3, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv14')
        conv14 = tf.nn.relu(conv_bn, name=scope.name)
    with tf.variable_scope('conv15') as scope:
        weights = tf.get_variable('weights', [3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(conv14, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv15')
        conv15 = tf.nn.relu(conv_bn, name=scope.name)
    with tf.variable_scope('conv16') as scope:
        weights = tf.get_variable('weights', [3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        conv = tf.nn.conv2d(conv15, weights, strides = [1, 1, 1, 1], padding = "SAME")
        conv_bn = BatchNorm(conv, is_training, scope='conv16')
        conv16 = tf.nn.relu(conv_bn, name=scope.name)
        conv16_flat = tf.reshape(conv16, [-1, 27*14*64])
    # 2 fully connected layers
    with tf.variable_scope('fc2'):
        weights = tf.get_variable('weights', [27*14*64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(conv16_flat, weights) + biases)
    with tf.variable_scope('fc3'):
        weights = tf.get_variable('weights', [64, 64], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        fc3 = tf.nn.relu(tf.matmul(fc2, weights) + biases)
    #dropout with the probility kept 'keep_prob'
	fc3_drop=tf.nn.dropout(fc3,keep_prob)
    # readout layer: output the probability of grasp success
    # we don't apply sigmoid here because
    # tf.nn.sigmoid_cross_entropy_with_logits accepts the unscaled logits
    # and performs the sigmoid internally for efficiency.
    with tf.variable_scope('out'):
        weights = tf.get_variable('weights', [64, 2], initializer=tf.truncated_normal_initializer(stddev = 0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', [2], initializer=tf.constant_initializer(0.1))
        logits = tf.matmul(fc3_drop, weights) + biases
    return logits

def loss(logits, labels):
    """Calculates the loss.
    Args:
        logits: logits from inference()
        labels: labels from inputs(). 1-D tensor of shape [batch_size]
    Returns:
        loss tensor of type float
    """
    # crossentropy H(p,q) = - sum{ p(x)log(q(x)) }, where p is the true distribution
    # http://stats.stackexchange.com/questions/167787/cross-entropy-cost-function-in-neural-network
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits), name='xentropy_mean')
    reg_losses=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant=0.01
    loss=loss+reg_constant*sum(reg_losses)
    return loss

def training(loss, learning_rate, global_step):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

    Returns:
    train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    #tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    #global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

