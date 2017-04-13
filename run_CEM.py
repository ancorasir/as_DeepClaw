import tensorflow as tf 
from training.Servoing import Servoing
import numpy as np 
from PIL import Image

sev = Servoing('/home/ancora-sirlab/xymeow/as_DeepClaw/training/checkpoint')
#box = (500, 350, 500+472, 350+472)

img_00 = np.array(Image.open('img00.jpg'))
img_01 = np.array(Image.open('img01.jpg'))
# M = 6, N = 64
print(sev.run(img_00, img_01, [-0.3, -0.2, 0, 0, 0, 0]))
