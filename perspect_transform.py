"""
This module calculate the perspecticve transfomation matrix from robot coordinate to image pixel coordinate

M_robotToImage * P_robot = P_image

| M11 M12 M13 |   | P1.x |   | w*P1'.x |
| M21 M22 M23 | * | P1.y | = | w*P1'.y |
| M31 M32 M33 |   | 1    |   | w*1     |

Calibration setup: camera is looking down faceing the same direction as robot
     Robot
       /\
      /  \
___________________

  point1   point4

  point2   point3
____________________
"""
# Xiaoyi's
#(-0.65 ~ -0.32, -0.2~0.23)

# new robot coordinates of the four corners of the bin
# (-0.736 ~ -0.265, -0.27 ~ 0.31)

import matplotlib.pyplot as plt
import numpy as np
import cv2


#points_image = np.float32([[822, 590], [822, 750], [1040, 750], [1040, 590]])
points_image = np.float32([[1040, 750], [1040, 590], [822, 590], [822, 750]])
points_robot = np.float32([[-0.4372, -0.1535], [-0.6117, -0.3164], [-0.3845, -0.5681], [-0.2066, -0.3971]])

M_robotToImage = cv2.getPerspectiveTransform(points_robot, points_image)
M_imageToRobot = cv2.getPerspectiveTransform(points_image, points_robot)

np.save('M_robotToImage', M_robotToImage)
np.save('M_imageToRobot', M_imageToRobot)

# test
M_robotToImage = np.load('M_robotToImage.npy')
M_imageToRobot = np.load('M_imageToRobot.npy')
np.matmul(M_robotToImage, np.float32([[-0.4372, -0.1535, 1]]).transpose())
np.matmul(M_imageToRobot, np.float32([[822, 590, 1]]).transpose())

# grasp boundary
new = np.matmul(M_imageToRobot, np.float32([[800, 560, 1], [794, 754, 1], [1051, 753, 1], [1048, 566, 1]]).transpose())
for i in range(4):
    new[:2,i]/new[2,i]

# grasp boundary in robot coordinate
#array([-0.4273897 , -0.09975769])
#array([-0.64421132, -0.28934557])
#array([-0.37600506, -0.5844999 ])
#array([-0.17146715, -0.38096566])

# generate random points within the grasp box
points_box = np.float32([[0, 0], [0, 1], [1, 1], [1, 0]])
points_robot = np.float32([[-0.4372, -0.1535], [-0.6117, -0.3164], [-0.3845, -0.5681], [-0.2066, -0.3971]])

M_boxToRobot = cv2.getPerspectiveTransform(points_box, points_robot)
np.matmul(M_boxToRobot, np.float32([[np.random.rand(), np.random.rand(), 1]]).transpose())
