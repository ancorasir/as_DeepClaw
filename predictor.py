"""
Code to run grasp detection given an image using the network learnt.
Copy and place the checkpoint files under './checkpoint' to load the trained network

Example run:

1. initialize predict model with pretrained weights

G = Predictor('./checkpoint_100index')

2. given input image, calculate best grasp location and theta, remember to translate the pixel location to robot grasp location.
   specify number of patches along horizontal axis, defaul = 10
   specify patch size by setting number of pixels along horizontal axis for a patch,   default = 360

image = Image.open('/home/ancora-sirlab/wanfang/cropped_image/hh.jpg').crop((300, 150, 1250, 1000))
location, theta = G.eval(image, num_patches_h, patch_pixels)

3. terminate the tensorflow session
G.close()
"""

import numpy as np
import tensorflow as tf
from graspNet import model
from PIL import Image

use_gpu_fraction = 0.8
NUM_THETAS = 18
SCALE_THETA = 100
INDICATORS = np.arange(1,NUM_THETAS+1).reshape([NUM_THETAS,1]) * SCALE_THETA

class Predictor:
    def __init__(self, checkpoint_path='./checkpoint'):
        self.checkpoint = tf.train.latest_checkpoint(checkpoint_path)

        # initialize prediction network for each patch
        self.images_batch = tf.placeholder(tf.float32, shape=[NUM_THETAS, 227, 227, 3])
        self.indicators_batch = tf.placeholder(tf.float32, shape=[NUM_THETAS, 1])

        self.M = model()
        self.M.initial_weights()
        logits = self.M.inference(self.images_batch, self.indicators_batch)
        self.y = tf.nn.softmax(logits)
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver = tf.train.Saver(variables)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = use_gpu_fraction
        self.sess = tf.Session(config = config)
        saver.restore(self.sess, self.checkpoint)

    def eval_theta(self, patches):
        # input images are grasp patches, for each patch, traverse all thetas
        NUM_PATCHES = patches.shape[0]

        best_theta = []
        best_probability = []
        for i in range(NUM_PATCHES):
            patch_thetas = np.tile( patches[i].reshape([1, 227, 227, 3]), [NUM_THETAS,1,1,1])
            y_value = self.sess.run(self.y, feed_dict={self.images_batch: patch_thetas, self.indicators_batch: INDICATORS})
            best_idx = np.argmax(y_value[:,1])
            best_theta.append(INDICATORS[best_idx][0])
            best_probability.append(y_value[best_idx,1])
        return np.array(best_theta)/SCALE_THETA, np.array(best_probability)

    def eval(self, image, position, num_patches_h = 10, patch_pixels = 360):
        # input images is full image, for each image, traverse locations to generate grasp patches and thetas
        patches, boxes = self.generate_patches(image, num_patches_h, patch_pixels)
        candidates_theta, candidates_probability= self.eval_theta(patches) #[number of patches]
        best_idx = np.argmax(candidates_probability)
        x_pixel =  sum(boxes[best_idx][0::2])/2
        y_pixel = sum(boxes[best_idx][1::2])/2
        theta = candidates_theta[best_idx] # theta here is the theta index ranging from 1 to 18
        # mapping pixel position to robot position, transform to pixel position in the original uncropped images first by plus 100
        x = ( 810 - 50 - (y_pixel + 150) )*0.46/480.0 - 0.73
        y = ( 1067 - 50 - (x_pixel + 300) )*0.6/615.0 - 0.25
        position[0] = x
        position[1] = y
        position.append((-3.14 + (theta-0.5)*(3.14/9)))
        return position  #[x, y, theta]

    def generate_patches(self, image, num_patches_w = 10, patch_pixels = 360):
        I_h, I_w, I_c = np.array(image).shape # width, height, channels

        patches = []
        boxes = []
        for i in range(0, I_w-patch_pixels, (I_w-patch_pixels)/num_patches_w):
            for j in range(0, I_h-patch_pixels, (I_w-patch_pixels)/num_patches_w):
                box = (i, j, i+patch_pixels, j+patch_pixels)
                patch = image
                patch = patch.crop(box).resize((227, 227), Image.ANTIALIAS)
                patches.append(np.array(patch))
                boxes.append(box)
        return np.array(patches), np.array(boxes) #[number of patches, 360, 360, 3], [[x_s, y_s, x_e, y_e]]

    def close(self):
        self.sess.close()

