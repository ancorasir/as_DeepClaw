import tensorflow as tf 
from CEM import CEM
import numpy as np 
from PIL import Image

cem = CEM('checkpoint')
box = (500, 350, 500+472, 350+472)

img_00 = np.array(Image.open('/home/ancora-sirlab/arcade_claw_test/2017-03-27 10:35:20/cam1_I_11_13_color.jpg').crop(box))
img_13 = np.array(Image.open('/home/ancora-sirlab/arcade_claw_test/2017-03-27 10:35:20/cam1_I_11_11_color.jpg').crop(box))
# M = 6, N = 64
print(cem.run(img_00, img_13, 6, 64, [-0.5, 0, 0, 0, 0, 0, 0]))
