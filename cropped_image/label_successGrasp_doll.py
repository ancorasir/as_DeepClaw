import os, re
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

rename_origin_folder = '/home/ancora-sirlab/wanfang/cropped_image/rename_origin_grasp_data'

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


data_ = []
for csvfile in sorted(glob.glob(rename_origin_folder+'/*.csv')):
    data_.append( pd.read_csv(csvfile) )

data = pd.concat(data_, ignore_index=True)
data_success = data.loc[data['success'] == 1].copy() # 2434 success grasp
num_success = len(data_success['success'])
data_success = data_success.reset_index(drop=True)
data_success['doll'] = pd.Series(np.empty((num_success)) * np.nan)


all_images = sorted(glob.glob('./croppedImage_jpg_success'+'/*.jpg'), key=numericalSort)

plt.ion()
for i in range(num_success):
    img_00 = mpimg.imread(all_images[3*i])
    img_1 = mpimg.imread(all_images[3*i+1])
    img_12 = mpimg.imread(all_images[3*i+2])
    img = np.concatenate((img_00,img_1,img_12),axis=1)
    plt.imshow(img)
    plt.pause(0.05)
    label = input("")
    data_success.loc[i,'doll'] = label

