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
import tensorflow as tf
import numpy as np
from fc_graspNet import FCGraspNet, GraspNet

class FCPredictor(object):
    def __init__(self, num_classes, checkpoint_path=None, width=1280, height=720):
        self.num_classes = num_classes
        self.num_thetas = int(num_classes/2)
        self.w = width
        self.h = height
        self.images_batch = tf.placeholder(tf.float32, shape=[None, self.h, self.w, 3])
        if checkpoint_path is not None:
            self.init_model(checkpoint_path)

    def init_model(self, checkpoint_path):
        self.model = FCGraspNet(self.num_classes)
        self.model.reload_network(checkpoint_path)
        # self.model.initialize_network('/home/h/DeepClawBenchmark/Functions/SoftGripper/checkpoint/Network9-1000-60')
        self.logits = self.model.inference(self.images_batch)
        print(self.logits.get_shape())
        logits_r = tf.reshape(self.logits, [1, self.logits.get_shape()[1].value, self.logits.get_shape()[2].value,
                                            int(self.logits.get_shape()[3].value / 2), 2])
        self.y = tf.nn.softmax(logits_r)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.sess = tf.Session(config=config)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        self.uv_range = [0, 0]

    def run(self, color_image):
        # img = image[:, :, ::-1]
        img = color_image.reshape(1, self.h, self.w, 3) - 164.0
        y_ = self.sess.run(self.y, feed_dict={self.images_batch: img})

        p_best = np.max(y_[0, self.uv_range[0]:15, self.uv_range[1]:28, :, 1], axis=2)
        location = np.where(p_best == p_best.max())
        best_u = 114 + location[1][0] * 32 + self.uv_range[1] * 32
        best_v = 114 + location[0][0] * 32 + self.uv_range[0] * 32
        local_best_theta = np.argmax(y_[0, self.uv_range[0]:15, self.uv_range[1]:24, :, 1], axis=2)
        global_best_theta = local_best_theta[location[0][0]][location[1][0]]
        best_prob = p_best.max()

        return y_, p_best, [best_u, best_v, - 1.57 + (global_best_theta + 0.5) * (1.57 / self.num_thetas) ]

class Predictor(object):
    def __init__(self, num_classes, checkpoint_path='./checkpoint'):
        self.num_classes = num_classes
        self.num_thetas = int(num_classes/2)
        self.checkpoint = tf.train.latest_checkpoint(checkpoint_path)

        # initialize prediction network for each patch
        self.images_batch = tf.placeholder(tf.float32, shape=[NUM_THETAS, 227, 227, 3])
        self.indicators_batch = tf.placeholder(tf.float32, shape=[NUM_THETAS, 1])

        self.model = GraspNet()
        self.model.initial_weights()
        logits = self.model.inference(self.images_batch, self.indicators_batch)
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

    def run(self, image, num_patches_h = 10, patch_pixels = 360):
        # input images is full image, for each image, traverse locations to generate grasp patches and thetas
        patches, boxes = self.generate_patches(image, num_patches_h, patch_pixels)
        candidates_theta, candidates_probability= self.eval_theta(patches) #[number of patches]

        best_idx = np.argmax(candidates_probability)
        best_u =  sum(boxes[best_idx][0::2])/2
        best_v = sum(boxes[best_idx][1::2])/2
        best_theta = candidates_theta[best_idx] # theta here is the theta index ranging from 1 to 18
        return [best_u, best_v, -3.14 + (best_theta - 0.5) * (3.14 / self.num_thetas)]

        # mapping pixel position to robot position, transform to pixel position in the original uncropped images first by plus 100
        # move this hand-eye transformation calculation out of the predictor
        # x = ( 810 - 50 - (y_pixel + 150) )*0.46/480.0 - 0.73
        # y = ( 1067 - 50 - (x_pixel + 300) )*0.6/615.0 - 0.25
        # position = [x, y, -3.14 + (theta-0.5)*(3.14/9)] #[x, y, theta]


    def close(self):
        self.sess.close()


if __name__ == '__main__':
    from PIL import Image, ImageDraw
    NUM_THETAS = 1
    p = FCPredictor(NUM_THETAS*2, './checkpoint_softgripper/Network1-1000-100')
    img = Image.open('./data_softgripper/img3/2500.jpg')
    img_arr = np.array(img)
    y_, p_best, grasp_pose = p.run(img_arr)
    draw = ImageDraw.Draw(img, 'RGBA')
    for i in range(15):
        for j in range(28):
            x =  114 + j*32
            y =  114 + i*32
            r = p_best[i][j] * 16
            draw.ellipse((x-r, y-r, x+r, y+r), (0, 0, 255, 125))
            # draw the grasp orientation if the model predict it
            if NUM_THETAS > 1:
                local_best_theta = np.argmax(y_[0, i, j, :, 1])
                local_best_theta = - 1.57 + (local_best_theta + 0.5) * (1.57 / NUM_THETAS)
                draw.line([(x - r*np.cos(local_best_theta), y + r*np.sin(local_best_theta)),
                           (x + r*np.cos(local_best_theta), y - r*np.sin(local_best_theta))],
                           fill=(255,255,255,125), width=10)
    img.show()
