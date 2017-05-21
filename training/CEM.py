# coding:utf-8
# author: Xiaoyi He

import googleGrasp as gg 
import tensorflow as tf 

import numpy as np 
import tf_utils

class CEM(object):
    """docstring for CEM"""
    def __init__(self, model_path):
        self.image = tf.placeholder(tf.float32, [1, 472*2, 472, 3], name='image')
        self.motions = tf.placeholder(tf.float32, [64, 3], name='motions')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.inference = gg.inference(self.image, self.motions, self.is_training)

        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session()
        self.sess.run(self.init_op)

        saver = tf.train.Saver()

        checkpoint = tf.train.latest_checkpoint(model_path)
        if checkpoint:
            saver.restore(self.sess, checkpoint)

    def run(self, image_00, image_01, M, N, position):
        # position: (-0.65 ~ -0.32, -0.2~0.23)
        mean = [0, 0, 0]
        cov = np.diagflat([1, 1, 1])

        images = np.concatenate((image_00, image_01), axis=0).reshape((1, 944, 472, 3))
        # google's work uses 3 iteration of optimization
        for iter in range(3):
            # sampling N grasp directions vt, shape = [N, 7]
            Xs = []
            # for _ in range(N):
            i = 0
            while i < N:
                X = np.random.multivariate_normal(mean, cov, 1)
                if -0.65 < X[0][0] < -0.32 and -0.2 < X[0][1] < 0.23 and -1.57 < X[0][2] < 1.57:
                # make sure the sample grasp in within the workspace of the robotic gripper
                #if (rotations<=180) and (gripper in workspace):
                    X[0][0] -= position[0] # x vector
                    X[0][1] -= position[1] # y vector
                    Xs.append(X[0])
                    i += 1
            Xs = np.array(Xs, dtype=np.float32)

            # select the 6 best grasp directions by inferring to the network
            # get images input from camera
            # load the trained network, ??need more work
            # performance = np.sum(Xs, axis=1) # for demonstration
            
            performance = self.inference.eval(session=self.sess, feed_dict={
                self.image: images,
                self.motions: Xs,
                self.is_training: False
                }).T[0]

            # Sort X by objective function values (in ascending order)
            best_idx = np.argsort(performance)[-M:]
            best_Xs = np.array(Xs)[best_idx,:]

            # Update parameters of distribution from the M best grasp directions
            mean = np.mean(best_Xs, axis=0)
            mean[0] += position[0]
            mean[1] += position[1]
            cov = np.cov(best_Xs, rowvar=0)
            #print(mean, cov)
        # use the optimized parameter to infer the best grasp direction
        Xs = []
        i = 0
        while i < N:
            X = np.random.multivariate_normal(mean, cov, 1)
            if -0.65 < X[0][0] < -0.32 and -0.2 < X[0][1] < 0.23 and -1.57 < X[0][-1] < 1.57:
            #if (rotations<=180) and (gripper in workspace)
                X[0][0] -= position[0] # x vector
                X[0][1] -= position[1] # y vector
                Xs.append(X[0])
                i += 1
        Xs = np.array(Xs)
        # selecting the best grasp directions
        performance = np.sum(Xs, axis=1)

        best_idx = np.argsort(performance)[-1:]
        position[0] += Xs[best_idx][0][0] 
        position[1] += Xs[best_idx][0][1]
        position.append(Xs[best_idx][0][2])
        return position
