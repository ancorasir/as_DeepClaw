import tensorflow as tf 
from Servoing import Servoing
import numpy as np 
from PIL import Image

servoing = Servoing('checkpoint')

box = (200, 0, 1280, 1080)
img_00 = np.array(Image.open('/home/ancora-sirlab/arcade_claw_test/test/2017-04-10 14:42:17/I_1_00_color_camB.jpg').crop(box).resize((472, 472), Image.ANTIALIAS))
img_01 = np.array(Image.open('/home/ancora-sirlab/arcade_claw_test/test/2017-04-10 14:42:17/I_1_01_color_camB.jpg').crop(box).resize((472, 472), Image.ANTIALIAS))
# M = 6, N = 64
images = np.concatenate([img_00,img_01],axis=0).reshape((1, 472*2, 472,3))
servoing.run(img_00, img_01, [-0.597, -0.027])

data=np.load('outfile.npz')
img_00_=data['img_00']
img_01_=data['img_01']
images_ = np.concatenate([img_00_,img_01_],axis=0).reshape((1, 472*2, 472,3))
servoing.performance(images,np.array([-0.011, -0.210, -2.592]).reshape((1,3)))
servoing.run(img_00_, img_01_, [-0.5, 0])

# not all input images lead to the same performance and p_empty
