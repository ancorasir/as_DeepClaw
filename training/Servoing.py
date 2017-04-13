# coding:utf-8
# author: Xiaoyi He

import googleGrasp as gg 
import tensorflow as tf 

import numpy as np 
import tf_utils

class Servoing(object):
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

    def cem(self, images, M, N, position):
        # TODO: make cem method private
        # position: (-0.65 ~ -0.32, -0.2~0.23)
        mean = [0, 0, 0]
        cov = np.diagflat([1, 1, 1])

        
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
        print(Xs)
        # selecting the best grasp directions
        performance = self.inference.eval(session=self.sess, feed_dict={
                self.image: images,
                self.motions: Xs,
                self.is_training: False
                }).T[0]
        print(performance)
        # performance = np.sum(Xs, axis=1)

        best_idx = np.argsort(performance)[-1:]
        position[0] += Xs[best_idx][0][0] 
        position[1] += Xs[best_idx][0][1]
        position.append(Xs[best_idx][0][2])

        return position, performance[best_idx]

    def run(self, image_00, image_01, position):
        '''
        public method of servoing mechanism
        '''
        images = np.concatenate((image_00, image_01), axis=0).reshape((1, 472*2, 472, 3))
        grasp = np.array([0, 0, 0], dtype=np.float32)
        grasp = np.array([grasp for _ in range(64)], dtype=np.float32)

        performance_grasp = self.inference.eval(session=self.sess, feed_dict={
                self.image: images,
                self.motions: grasp,
                self.is_training: False
                }).T[0][0]
        #print('position', position)
        new_position, performance = self.cem(images, 6, 64, position)
        #print('new position', new_position)
        print('pg=', performance_grasp, 'p0=', performance)
        p = performance_grasp / performance[0]
        flag = 0
        print('p=', p)
        if p > 0.9:
            # grasp
            flag = 2
            new_position.append(flag)
            return new_position
        elif p < 0.5:
            # stop and quit
            flag = 0
            new_position.append(flag)
            return new_position
        else: 
            # 0.5 < p < 0.9, execute motion
            flag = 1
            new_position.append(flag)
            return new_position




