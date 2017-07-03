import os, sys
import time
from image_saver_C import ImageSaver

data_path = './' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '/'
os.mkdir(data_path)

img_saver = ImageSaver()
iteration = 0
img_saver.kinect_saver(data_path + 'I_' + str(iteration))
