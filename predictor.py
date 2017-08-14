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

# patch size in pixels
PATCH_PIXELS = 250
INDICATORS = np.arange(1,NUM_THETAS+1).reshape([NUM_THETAS,1]) * SCALE_THETA

M_imageToRobot = np.load('M_imageToRobot.npy')

class Predictor:
    def __init__(self, checkpoint_path='./checkpoint'):
        self.checkpoint = tf.train.latest_checkpoint(checkpoint_path)

        # initialize prediction network for each patch
        self.images_batch = tf.placeholder(tf.float32, shape=[NUM_THETAS, 227, 227, 3])
        self.indicators_batch = tf.placeholder(tf.float32, shape=[NUM_THETAS, 3])

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

    def eval(self, image, position, num_patches_w = 10, patch_pixels = PATCH_PIXELS):
        I_h, I_w, I_c = np.array(image).shape # width, height, channels

        boxes = []
        best_theta = []
        best_probability = []
        for i in range(0, I_w-patch_pixels, (I_w-patch_pixels)/num_patches_w):
            for j in range(0, I_h-patch_pixels, (I_w-patch_pixels)/num_patches_w):
                box = (i, j, i+patch_pixels, j+patch_pixels)
                x_pixel = i + patch_pixels/2
                y_pixel = j + patch_pixels/2
                location = np.array([x_pixel*(277.0/465), y_pixel*(277.0/410)]).astype(int)
                patch = np.array(image.resize((227, 227), Image.ANTIALIAS))
                patch_thetas = np.tile(patch.reshape([1, 227, 227, 3]), [NUM_THETAS,1,1,1])
                location_thetas = np.tile( location.reshape([1, 2]), [NUM_THETAS,1]) #[NUM_THETAS,2]
                indicators = np.concatenate([location_thetas, INDICATORS], axis=1) #[NUM_THETAS,3]
                y_value = self.sess.run(self.y, feed_dict={self.images_batch: patch_thetas, self.indicators_batch: INDICATORS})
                best_idx = np.argmax(y_value[:,1])
                best_theta.append(INDICATORS[best_idx][0])
                best_probability.append(y_value[best_idx,1])
                boxes.append(box)
        candidates_theta = np.array(best_theta)/SCALE_THETA
        candidates_probability = np.array(best_probability)

        best_idx = np.argmax(candidates_probability)
        x_pixel =  sum(boxes[best_idx][0::2])/2
        y_pixel = sum(boxes[best_idx][1::2])/2
        theta = candidates_theta[best_idx] # theta here is the theta index ranging from 1 to 18

        new = np.matmul(M_imageToRobot, np.float32([[x_pixel+700, y_pixel+465, 1]]).transpose()) #[3,1]
        new_xy = new[:2,0]/new[2,0]
        position[0] = new_xy[0]
        position[1] = new_xy[1]

        theta_calibration = 0
        rotation = (-3.1415 + (theta-0.5)*(3.1415/9)) + theta_calibration
        if rotation<-3.1415:
            rotation += 2*3.1415
        position.append(rotation)
        return position  #[x, y, theta]

    def close(self):
        self.sess.close()

