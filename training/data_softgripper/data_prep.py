import os, re
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import glob
import shutil

rename_origin_folder = './img3'
NUM_THETAS = 18

# read grasp data prepared by label_graspCenter_isEmpty.py
#data = pd.read_csv('./data_beta_grasp_data.csv')

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# grasp data: 828 success grasps among 2000 grasps, 41.4%
data_ = []
for csvfile in sorted(glob.glob('./csv/*.csv')):
    data_.append( pd.read_csv(csvfile) )

data_2000 = pd.concat(data_, ignore_index=True)
data_2000.to_csv('./grasp_data_2000.csv')

# grasp data: 270 success grasps among 1000 grasps, with less objects in the bin
data_ = []
for csvfile in sorted(glob.glob('./csv2/*.csv')):
    data_.append( pd.read_csv(csvfile) )

for csvfile in sorted(glob.glob('./csv3/*.csv')):
    data_.append( pd.read_csv(csvfile) )

data_1000 = pd.concat(data_, ignore_index=True)
data_1000.to_csv('./grasp_data_1000.csv')
data_1000.loc[data['label'] == 1]

# grasp data: select 750 fail and 250 success from 2000 grasps
for csvfile in sorted(glob.glob('./csv/*.csv')):
    data_.append( pd.read_csv(csvfile) )

data_fail = data.loc[data['label'] == 0].copy()
data_success = data.loc[data['label'] == 1].copy()

data_fail = data_fail.sample(n=750)
data_success = data_success.sample(n=250)
data_1000_ = pd.concat([data_fail,data_success])
data_1000_sorted = data_1000_.sort_index()

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def crop_image(src_folder, dis_folder):
    """
    Crop image_00 and image_1, save cropped images in jpg format
    """
    if not os.path.isdir(dis_folder):
        print('dis folder not exist! now creating a new one ...')
        os.mkdir(dis_folder)
    # list all images
    all_images = sorted(glob.glob(src_folder+'/*.jpg'), key=numericalSort)
    for i in range(len(all_images)):
        # determine the location of crop box
        u = data['u'][i]
        v = data['v'][i]
        angle = data['angel'][i]
        shft=125
        crop_box = (u-shft,v-shft,u+shft,v+shft)
        # save cropped image in jpg
        image_00 = Image.open(all_images[i]).crop(crop_box)
        image_00.save(dis_folder + "/%s_cropped.jpg"%all_images[i].split('/')[-1][:-4])

def tf_writer(src_folder, dis_folder):
    """
    change the datas under src_folder into TFRecord fomat
    Args:
        src_folder: source folder with image and csv datas
        dis_folder: directory to save tfrecord file
    """
    if not os.path.isdir(dis_folder):
        print('dis folder not exist! now creating a new one ...')
        os.mkdir(dis_folder)
    # list all images
    all_images = sorted(glob.glob(src_folder+'/*.jpg'), key=numericalSort)
    # writer = tf.python_io.TFRecordWriter(dis_folder+'/croppedImage_500.tfrecord')
    writer = tf.python_io.TFRecordWriter(dis_folder+'/croppedImage_500_1.tfrecord')
    for i in range(500):
        # if i%3000==0:
        #     print('Writing %s.tfrecord'%(i/1000))
        #     if i!=0:
        #         writer.close()
        #     writer = tf.python_io.TFRecordWriter(dis_folder+'/croppedImage_%s'%(i/1000) + '.tfrecord')
        # read cropped image in jpg
        img = np.array(Image.open(all_images[i+2500]).resize((227, 227), Image.ANTIALIAS)).tobytes()
        # save image and graps data into tfrecord files
        example = tf.train.Example(features=tf.train.Features(
            feature={
            'name': _bytes_feature(all_images[i+2500].split('/')[-1]),
            'img': _bytes_feature(img),
            'angle': _floats_feature([data_1000['angel'][i+500]]),
            'label': _floats_feature([data_1000['label'][i+500]]),
            }))
        writer.write(example.SerializeToString())
    writer.close()
    return

def tf_writer_1(src_folder, dis_folder):
    """
    change the datas under src_folder into TFRecord fomat
    Args:
        src_folder: source folder with image and csv datas
        dis_folder: directory to save tfrecord file
    """
    if not os.path.isdir(dis_folder):
        print('dis folder not exist! now creating a new one ...')
        os.mkdir(dis_folder)
    # list all images
    all_images = sorted(glob.glob(src_folder+'/*.jpg'), key=numericalSort)
    writer = tf.python_io.TFRecordWriter(dis_folder+'/croppedImage_selected1000.tfrecord')
    for i in data_1000_sorted.index:
        img = np.array(Image.open(all_images[i]).resize((227, 227), Image.ANTIALIAS)).tobytes()
        # save image and graps data into tfrecord files
        example = tf.train.Example(features=tf.train.Features(
            feature={
            'name': _bytes_feature(all_images[i].split('/')[-1]),
            'img': _bytes_feature(img),
            'angle': _floats_feature([data_2000['angel'][i]]),
            'label': _floats_feature([data_2000['label'][i]]),
            }))
        writer.write(example.SerializeToString())
    writer.close()
    return

def image_mean(src_folder):
    all_images = sorted(glob.glob(src_folder+'/*.jpg'), key=numericalSort)
    s = np.array([0.0,0.0,0.0])
    for i in range(len(all_images)):
        # read cropped image in jpg
        img_00 = np.array(Image.open(all_images[i]))
        s += [np.mean(img_00[:,:,j]) for j in range(3)]
    m = s/len(all_images)
    print(m)
    # m = [ 164.29138336  164.68029373  163.59514228]

crop_image("./img3", './croppedImg')
tf_writer('./croppedImg', './')
image_mean('./croppedImg') # m = [131.12977262 122.84692178 112.12587453]

# write selected 1000 images from the 0 ~ 1999 images, 750 fail and 250 success
tf_writer_1('./croppedImg', './')
