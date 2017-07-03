import os, re
import numpy as np
import pandas as pd
import glob
from PIL import Image

rename_origin_folder = '/home/ancora-sirlab/wanfang/cropped_image/rename_origin_grasp_data_new'

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

data_ = []
for csvfile in sorted(glob.glob(rename_origin_folder+'/*.csv')):
    data_.append( pd.read_csv(csvfile) )

data = pd.concat(data_, ignore_index=True)
total_grasp_num = len(data['success'])
#data['isDoll'] = pd.Series(np.empty((total_grasp_num)) * np.nan)
data['isDoll'] = pd.Series(np.zeros(total_grasp_num))

all_image_00 = sorted(glob.glob('./croppedImage_jpg_new'+'/*.jpg'), key=numericalSort)

grasp_box = [90, 90, 270, 270]
threshhold = 41
for i in range(0, total_grasp_num):
    # cropped image shape [360, 360,3]
    img = Image.open(all_image_00[i])
    img_grasp = img.crop(grasp_box)
    diff = np.abs(np.array(img_grasp) - np.ones([180, 180, 3])*255)
    #diff = np.abs(np.array(img_grasp) - np.array(img_bench.crop(grasp_box)))
    diff.sort()
    if np.mean(diff[-180*180:])>=41:
        data.loc[i,'isDoll'] = 1

# set 'success' to 0 where there is no doll in the grasp box
data.loc[(data.success == 1) & (data.isDoll == 0),'success']
data.loc[(data.success == 1) & (data.isDoll == 0),'success'] = 1

data.to_csv('./data_origin_grasp_data.csv')

# some of the strange data: fail labeled as success (388,4543), doll moves to the center before grasp (353,377), 1360 success
# threshhold = 45, i= [353,377,388,1360:41.3,4543,6116:41.2,6487:42.4,8621,9266]
# 20/06/2017, threshhold = 41, i = [377, 4543, 6090]

img_bench=Image.open('./croppedImage_jpg/2017-05-07 11:52:37_I_37_00_color_camB.jpg')
np.mean(np.mean(np.array(img_bench.crop([90, 90, 270, 270])), axis=0),axis=0)
# [206, 215, 222]
np.std(np.array(img.crop([90, 90, 250, 250])))
# 18.8

