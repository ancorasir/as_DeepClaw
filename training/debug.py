# import tf_utils
#
# import time
# import os
#
# from graspNet import model as grasp_net
#
# import tensorflow as tf
# import tensorflow.contrib.framework as tcf
#
# # 40, 10min/epoch;
# batch_size = 100
# num_epochs = 15
# #starter_learning_rate = 0.05
# use_gpu_fraction = 1
#
# max_learning_rate = 0.001
# min_learning_rate = 0.0001
# decay_speed = 1000
#
# checkpoint_path = './checkpoint'
# summary_path = './summary'
# data_path = '/home/bionicdl/git-projects-py2/as_DeepClaw/training/croppedImage_tfrecord'
# ckpt_file = tf.train.latest_checkpoint(checkpoint_path)
# reader = tf.train.NewCheckpointReader(ckpt_file)
#
# # Create empty weight object.
# weights = {}
#
# # Read/generate weight/bias variable names.
# ckpt_vars = tcf.list_variables(ckpt_file)
# full_var_names = []
# short_names = []
# for variable, shape in ckpt_vars:
#     full_var_names.append(variable)
#     short_names.append(variable.split("/")[-1])
#
# # Load variables.
# for full_var_name, short_name in zip(full_var_names, short_names):
#     weights[short_name] = tf.Variable(reader.get_tensor(full_var_name), name=full_var_name)
#
# _build_conv_layer
# _build_fc_layer
#
# checkpoint = tf.train.latest_checkpoint(checkpoint_path)

# initialize prediction network for each patch
from fc_graspNet import fcmodel
import cv2
from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

images_batch = tf.placeholder(tf.float32, shape=[None, 720, 1280, 3])

M = fcmodel(18)
M.initialize_network('./checkpoint_softgripper/Network9-1000-40')

logits = M.inference(images_batch)
logits_r = tf.reshape(logits, [1,logits.get_shape()[1].value,logits.get_shape()[2].value,logits.get_shape()[3].value/2,2])
y = tf.nn.softmax(logits_r)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config = config)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

# path = './data_softgripper/croppedImg/1_cropped.jpg'
# img = cv2.resize(img,(227,227)) - 164.0
for i in range(1,2):
    path = './data_softgripper/img/%s.jpg'%(i)
    # path = './test_images/170523-022917_I_664_00_color_camA.jpg'
    img = cv2.imread(path)[:,:,::-1]
    h = img.shape[0]
    w = img.shape[1]
    img = img.reshape(1, h, w, 3) - 164.0

    logits_ = sess.run(logits, feed_dict={images_batch:img})
    y_ = sess.run(y, feed_dict={images_batch:img})

    p_best= np.max(y_[0,:,:,:,1],axis=2)
    theta_best = np.argmax(y_[0,:,:,:,1],axis=2)
    I = Image.open(path)
    draw = ImageDraw.Draw(I, 'RGBA')
    for i in range(p_best.shape[0]):
        for j in range(p_best.shape[1]):
            u = 114+j*32
            v = 114+i*32
            r = p_best[i,j] * 16
            draw.ellipse((u-r, v-r, u+r, v+r), (0, 0, 255, 125))
            if p_best[i,j]>0.5:
                # tranasform theta from index to [-pi, pi]
                # initial grasp plate is horizontal, clockwise rotation is positive 0 ~ pi, anti-clockwise is negative
                best_theta = -3.14/2 + (theta_best[i,j]+0.5)*(3.14/9)
                draw.line([(u-r*np.cos(best_theta),v+r*np.sin(best_theta)),
                       (u+r*np.cos(best_theta), v-r*np.sin(best_theta))], fill=(255,255,255,125), width=10)

    I.save("test_p_9-1000-40.png")
    # I.show()
