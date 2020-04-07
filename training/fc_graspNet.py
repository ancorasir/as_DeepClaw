from __future__ import print_function

import numpy as np
import os
import time
import tensorflow as tf
import tensorflow.contrib.framework as tcf

class FCGraspNet(object):
    # fully convolutional grasp net, modified from GraspNet by converting all the fully connected layer to 1x1 convolutional layer
    def __init__(self, num_classes):
        # initialte the model with the number of classes
        self.num_classes = num_classes

    def initial_weights(self, weight_file=None):
        if weight_file:
            # Load what you want the initialisation to be, weight_file is the path './bvlc_alexnet.npy'
            print('Loading weights from {0}'.format(weight_file))
            net_data = np.load(weight_file).item()
            conv1W_init = net_data["conv1"][0]
            conv1b_init = net_data["conv1"][1]
            conv2W_init = net_data["conv2"][0]
            conv2b_init = net_data["conv2"][1]
            conv3W_init = net_data["conv3"][0]
            conv3b_init = net_data["conv3"][1]
            conv4W_init = net_data["conv4"][0]
            conv4b_init = net_data["conv4"][1]
            conv5W_init = net_data["conv5"][0]
            conv5b_init = net_data["conv5"][1]
            fc6W_init = tf.truncated_normal([9216, 4096], stddev=    0.01)
            fc6b_init = tf.constant(0.1, shape=[4096])
            fc7W_init = tf.truncated_normal([4096, 1024], stddev=    0.01)
            fc7b_init = tf.constant(0.1, shape=[1024])
            fc8W_init = tf.truncated_normal([1024,self.num_classes], stddev=    0.01)
            fc8b_init = tf.constant(0.1, shape=[self.num_classes])
        else:
            conv1W_init = tf.truncated_normal([11, 11, 3, 96], stddev=    0.01)
            conv1b_init = tf.constant(0.1, shape=[96])
            conv2W_init = tf.truncated_normal([5, 5, 48, 256], stddev=    0.01)
            conv2b_init = tf.constant(0.1, shape=[256])
            conv3W_init = tf.truncated_normal([3, 3, 256, 384], stddev=    0.01)
            conv3b_init = tf.constant(0.1, shape=[384])
            conv4W_init = tf.truncated_normal([3, 3, 192, 384], stddev=    0.01)
            conv4b_init = tf.constant(0.1, shape=[384])
            conv5W_init = tf.truncated_normal([3, 3, 192, 256], stddev=    0.01)
            conv5b_init = tf.constant(0.1, shape=[256])
            fc6W_init = tf.truncated_normal([9216, 4096], stddev=    0.01)
            fc6b_init = tf.constant(0.1, shape=[4096])
            fc7W_init = tf.truncated_normal([4096, 1024], stddev=    0.01)
            fc7b_init = tf.constant(0.1, shape=[1024])
            fc8W_init = tf.truncated_normal([1024,self.num_classes], stddev=    0.01)
            fc8b_init = tf.constant(0.1, shape=[self.num_classes])

        self.conv1W = tf.Variable(conv1W_init,trainable=False, name='conv1W')
        self.conv1b = tf.Variable(conv1b_init,trainable=False, name='conv1b')
        self.conv2W = tf.Variable(conv2W_init,trainable=False, name='conv2W')
        self.conv2b = tf.Variable(conv2b_init,trainable=False, name='conv2b')
        self.conv3W = tf.Variable(conv3W_init,trainable=False, name='conv3W')
        self.conv3b = tf.Variable(conv3b_init,trainable=False, name='conv3b')
        self.conv4W = tf.Variable(conv4W_init,trainable=False, name='conv4W')
        self.conv4b = tf.Variable(conv4b_init,trainable=False, name='conv4b')
        self.conv5W = tf.Variable(conv5W_init,trainable=False, name='conv5W')
        self.conv5b = tf.Variable(conv5b_init,trainable=False, name='conv5b')
        self.fc6W = tf.Variable(fc6W_init, name='fc6W')
        self.fc6b = tf.Variable(fc6b_init, name='fc6b')
        self.fc7W = tf.Variable(fc7W_init, name='fc7W')
        self.fc7b = tf.Variable(fc7b_init, name='fc7b')
        self.fc8W = tf.Variable(fc8W_init, name='fc8W')
        self.fc8b = tf.Variable(fc8b_init, name='fc8b')
        self.dropfc6 = 1
        self.dropfc7 = 1

    def reload_network(self, ckpt_file=None):
        if ckpt_file:
            # Load what you want the initialisation to be, weight_file is the path './bvlc_alexnet.npy'
            # print('Loading weights from {0}'.format(weight_file))
            reader = tf.train.NewCheckpointReader(ckpt_file)

            # Create empty weight object.
            weights = {}

            # Read/generate weight/bias variable names.
            ckpt_vars = tcf.list_variables(ckpt_file)
            full_var_names = []
            short_names = []
            for variable, shape in ckpt_vars:
                full_var_names.append(variable)
                short_names.append(variable.split("/")[-1])

            # Load variables.
            for full_var_name, short_name in zip(full_var_names, short_names):
                weights[short_name] = tf.Variable(reader.get_tensor(full_var_name), name=full_var_name)

            self.conv1W = weights["conv1W"]
            self.conv1b = weights["conv1b"]
            self.conv2W = weights["conv2W"]
            self.conv2b = weights["conv2b"]
            self.conv3W = weights["conv3W"]
            self.conv3b = weights["conv3b"]
            self.conv4W = weights["conv4W"]
            self.conv4b = weights["conv4b"]
            self.conv5W = weights["conv5W"]
            self.conv5b = weights["conv5b"]
            self.fc6W = weights["fc6W"]
            self.fc6b = weights["fc6b"]
            self.fc7W = weights["fc7W"]
            self.fc7b = weights["fc7b"]
            self.fc8W = weights["fc8W"]
            self.fc8b = weights["fc8b"]
        else:
            conv1W_init = tf.truncated_normal([11, 11, 3, 96], stddev=    0.01)
            conv1b_init = tf.constant(0.1, shape=[96])
            conv2W_init = tf.truncated_normal([5, 5, 48, 256], stddev=    0.01)
            conv2b_init = tf.constant(0.1, shape=[256])
            conv3W_init = tf.truncated_normal([3, 3, 256, 384], stddev=    0.01)
            conv3b_init = tf.constant(0.1, shape=[384])
            conv4W_init = tf.truncated_normal([3, 3, 192, 384], stddev=    0.01)
            conv4b_init = tf.constant(0.1, shape=[384])
            conv5W_init = tf.truncated_normal([3, 3, 192, 256], stddev=    0.01)
            conv5b_init = tf.constant(0.1, shape=[256])
            fc6W_init = tf.truncated_normal([9216, 4096], stddev=    0.01)
            fc6b_init = tf.constant(0.1, shape=[4096])
            fc7W_init = tf.truncated_normal([4096, 1024], stddev=    0.01)
            fc7b_init = tf.constant(0.1, shape=[1024])
            fc8W_init = tf.truncated_normal([1024,self.num_classes], stddev=    0.01)
            fc8b_init = tf.constant(0.1, shape=[self.num_classes])
            self.conv1W = tf.Variable(conv1W_init,trainable=False, name='conv1W')
            self.conv1b = tf.Variable(conv1b_init,trainable=False, name='conv1b')
            self.conv2W = tf.Variable(conv2W_init,trainable=False, name='conv2W')
            self.conv2b = tf.Variable(conv2b_init,trainable=False, name='conv2b')
            self.conv3W = tf.Variable(conv3W_init,trainable=False, name='conv3W')
            self.conv3b = tf.Variable(conv3b_init,trainable=False, name='conv3b')
            self.conv4W = tf.Variable(conv4W_init,trainable=False, name='conv4W')
            self.conv4b = tf.Variable(conv4b_init,trainable=False, name='conv4b')
            self.conv5W = tf.Variable(conv5W_init,trainable=False, name='conv5W')
            self.conv5b = tf.Variable(conv5b_init,trainable=False, name='conv5b')
            self.fc6W = tf.Variable(fc6W_init, name='fc6W')
            self.fc6b = tf.Variable(fc6b_init, name='fc6b')
            self.fc7W = tf.Variable(fc7W_init, name='fc7W')
            self.fc7b = tf.Variable(fc7b_init, name='fc7b')
            self.fc8W = tf.Variable(fc8W_init, name='fc8W')
            self.fc8b = tf.Variable(fc8b_init, name='fc8b')

        self.dropfc6 = 1
        self.dropfc7 = 1

    def inference(self, image_batch):
        #conv1
        #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
        conv1_in = conv(image_batch, self.conv1W, self.conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        self.conv1_in = conv1_in
        conv1 = tf.nn.relu(conv1_in)
        #lrn1
        #lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
        #maxpool1
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        #conv2
        #conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv2_in = conv(maxpool1, self.conv2W, self.conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)
        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
        #maxpool2
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        #conv3
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3_in = conv(maxpool2, self.conv3W, self.conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)
        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4_in = conv(conv3, self.conv4W, self.conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)
        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5_in = conv(conv4, self.conv5W, self.conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)
        #maxpool5
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding) #[6,6,256]
        self.maxpool5 = maxpool5

        # fc6 convert to fully convolutional layer
        # fc(4096, name='fc6')
        k_h = 6; k_w = 6; c_o = 4096; s_h = 1; s_w = 1; group = 1
        self.conv6W = tf.reshape(self.fc6W, [6,6,256,4096])
        conv6_in = conv(maxpool5, self.conv6W , self.fc6b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=group )
        conv6 = tf.nn.relu(conv6_in)
        self.conv6 = conv6
        # fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]),
        #         self.fc6W, self.fc6b)
        #dropout
        drop6 = tf.nn.dropout(conv6,self.dropfc6)

        #fc7
        #fc(1024, name='fc7')
        k_h = 1; k_w = 1; c_o = 1024; s_h = 1; s_w = 1; group = 1
        self.conv7W = tf.reshape(self.fc7W, [1,1,4096,1024])
        conv7_in = conv(drop6, self.conv7W , self.fc7b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=group )
        conv7 = tf.nn.relu(conv7_in)
        self.conv7 = conv7
        # fc7 = tf.nn.relu_layer(drop6, self.fc7W, self.fc7b)
        #dropout
        drop7 = tf.nn.dropout(conv7,self.dropfc7)

        #fc8
        #fc(OUTPUT_SIZE, relu=False, name='fc8')
        k_h = 1; k_w = 1; c_o = self.num_classes; s_h = 1; s_w = 1; group = 1
        self.conv8W = tf.reshape(self.fc8W, [1,1,1024,self.num_classes])
        logits = conv(drop7, self.conv8W , self.fc8b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=group )
        # logits = tf.nn.xw_plus_b(drop7, self.fc8W, self.fc8b)
        return logits

class GraspNet(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def initial_weights(self, weight_file=None):
        if weight_file:
            # Load what you want the initialisation to be, weight_file is the path './bvlc_alexnet.npy'
            print('Loading weights from {0}'.format(weight_file))
            net_data = np.load(weight_file).item()
            conv1W_init = net_data["conv1"][0]
            conv1b_init = net_data["conv1"][1]
            conv2W_init = net_data["conv2"][0]
            conv2b_init = net_data["conv2"][1]
            conv3W_init = net_data["conv3"][0]
            conv3b_init = net_data["conv3"][1]
            conv4W_init = net_data["conv4"][0]
            conv4b_init = net_data["conv4"][1]
            conv5W_init = net_data["conv5"][0]
            conv5b_init = net_data["conv5"][1]
            fc1W_init = tf.truncated_normal([1, 256], stddev=    0.01)
            fc1b_init = tf.constant(0.1, shape=[256])
            fc6W_init = tf.truncated_normal([9216, 4096], stddev=    0.01)
            fc6b_init = tf.constant(0.1, shape=[4096])
            fc7W_init = tf.truncated_normal([4096, 1024], stddev=    0.01)
            fc7b_init = tf.constant(0.1, shape=[1024])
            fc8W_init = tf.truncated_normal([1024,self.num_classes], stddev=    0.01)
            fc8b_init = tf.constant(0.1, shape=[self.num_classes])
        else:
            conv1W_init = tf.truncated_normal([11, 11, 3, 96], stddev=    0.01)
            conv1b_init = tf.constant(0.1, shape=[96])
            conv2W_init = tf.truncated_normal([5, 5, 48, 256], stddev=    0.01)
            conv2b_init = tf.constant(0.1, shape=[256])
            conv3W_init = tf.truncated_normal([3, 3, 256, 384], stddev=    0.01)
            conv3b_init = tf.constant(0.1, shape=[384])
            conv4W_init = tf.truncated_normal([3, 3, 192, 384], stddev=    0.01)
            conv4b_init = tf.constant(0.1, shape=[384])
            conv5W_init = tf.truncated_normal([3, 3, 192, 256], stddev=    0.01)
            conv5b_init = tf.constant(0.1, shape=[256])
            fc1W_init = tf.truncated_normal([1, 256], stddev=    0.01)
            fc1b_init = tf.constant(0.1, shape=[256])
            fc6W_init = tf.truncated_normal([9216, 4096], stddev=    0.01)
            fc6b_init = tf.constant(0.1, shape=[4096])
            fc7W_init = tf.truncated_normal([4096, 1024], stddev=    0.01)
            fc7b_init = tf.constant(0.1, shape=[1024])
            fc8W_init = tf.truncated_normal([1024,self.num_classes], stddev=    0.01)
            fc8b_init = tf.constant(0.1, shape=[self.num_classes])

        self.conv1W = tf.Variable(conv1W_init,trainable=False, name='conv1W')
        self.conv1b = tf.Variable(conv1b_init,trainable=False, name='conv1b')
        self.conv2W = tf.Variable(conv2W_init,trainable=False, name='conv2W')
        self.conv2b = tf.Variable(conv2b_init,trainable=False, name='conv2b')
        self.conv3W = tf.Variable(conv3W_init,trainable=False, name='conv3W')
        self.conv3b = tf.Variable(conv3b_init,trainable=False, name='conv3b')
        self.conv4W = tf.Variable(conv4W_init,trainable=False, name='conv4W')
        self.conv4b = tf.Variable(conv4b_init,trainable=False, name='conv4b')
        self.conv5W = tf.Variable(conv5W_init,trainable=False, name='conv5W')
        self.conv5b = tf.Variable(conv5b_init,trainable=False, name='conv5b')
        self.fc1W = tf.Variable(fc1W_init, name='fc1W')
        self.fc1b = tf.Variable(fc1b_init, name='fc1b')
        self.fc6W = tf.Variable(fc6W_init, name='fc6W')
        self.fc6b = tf.Variable(fc6b_init, name='fc6b')
        self.fc7W = tf.Variable(fc7W_init, name='fc7W')
        self.fc7b = tf.Variable(fc7b_init, name='fc7b')
        self.fc8W = tf.Variable(fc8W_init, name='fc8W')
        self.fc8b = tf.Variable(fc8b_init, name='fc8b')
        self.dropfc6 = 1
        self.dropfc7 = 1

    def inference(self, image_batch, indicators_batch):
        #conv1
        #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
        conv1_in = conv(image_batch, self.conv1W, self.conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        self.conv1_in = conv1_in
        conv1 = tf.nn.relu(conv1_in)
        #lrn1
        #lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
        #maxpool1
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        #conv2
        #conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv2_in = conv(maxpool1, self.conv2W, self.conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)
        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
        #maxpool2
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        #conv3
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3_in = conv(maxpool2, self.conv3W, self.conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)
        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4_in = conv(conv3, self.conv4W, self.conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)
        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5_in = conv(conv4, self.conv5W, self.conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)
        #maxpool5
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding) #[6,6,256]

        # Add indicator here to maxpool5
        fc1 = tf.nn.relu_layer(indicators_batch, self.fc1W, self.fc1b)
        fc1 = tf.reshape(fc1, shape=[-1, 1, 1, 256])
        maxpool5 = tf.add(tf.tile(fc1, [1, 6, 6, 1]), maxpool5)

        #fc6
        #fc(4096, name='fc6')
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]),
                self.fc6W, self.fc6b)
        #dropout
        drop6 = tf.nn.dropout(fc6,self.dropfc6)
        #fc7
        #fc(1024, name='fc7')
        fc7 = tf.nn.relu_layer(drop6, self.fc7W, self.fc7b)
        #dropout
        drop7 = tf.nn.dropout(fc7,self.dropfc7)
        #fc8
        #fc(OUTPUT_SIZE, relu=False, name='fc8')
        logits = tf.nn.xw_plus_b(drop7, self.fc8W, self.fc8b)
        #Debug stuff
        self.fc7 = fc7
        self.fc6 = fc6
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.conv5 = conv5
        self.fc1 = fc1
        #End Debug
        return logits

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    if group==1:
        conv = convolve(input, kernel)
    else:
        #input_groups = tf.split(3, group, input)
        input_groups = tf.split(input, group, 3)
        #kernel_groups = tf.split(3, group, kernel)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

# import keras
# from keras.datasets import cifar10
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# class SimpleModel:
#     def __init__(self, NUM_CLASSES):
#         self.NUM_CLASSES = NUM_CLASSES
#
#     def inference(self, image_batch):
#         model = Sequential()
#         model.add(Conv2D(32, (3, 3), padding='same', input_shape=image_batch.shape[1:]))
#         model.add(Activation('relu'))
#
#         model.add(Conv2D(32, (3, 3)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.25))
#
#         model.add(Conv2D(64, (3, 3), padding='same'))
#         model.add(Activation('relu'))
#
#         model.add(Conv2D(64, (3, 3)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.25))
#
#         model.add(Flatten())
#         model.add(Dense(512))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(num_classes))
#         model.add(Activation('softmax'))
