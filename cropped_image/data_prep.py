import os, re
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import glob
import shutil

rename_origin_folder = '/home/ancora-sirlab/wanfang/cropped_image/rename_origin_grasp_data'
NUM_THETAS = 18

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

data = pd.read_csv('./data_origin_grasp_data.csv')

# crop image_00 and image_1, save cropped  images in jpg 
def crop_image(src_folder, dis_folder):
    if not os.path.isdir(dis_folder):
        print('dis folder not exist! now creating a new one ...')
        os.mkdir(dis_folder)

    # list all images
    all_images = sorted(glob.glob(src_folder+'/*.jpg'), key=numericalSort)

    for i in range(len(all_images)/2):
        # determine the location of crop box
        coord = eval(data['move_1'][i][1:])
        angle = data['rotate_angle'][i]
        x = coord[0]
        y = coord[1]
        lftpix = int(1067-615*(y+0.25)/0.6)-50
        uppix = int(810-480*(x+0.73)/0.46)-50
        shft=180
        crop_box = (lftpix-shft,uppix-shft,lftpix+shft,uppix+shft)
      
        # save cropped image in jpg
        image_00 = Image.open(all_images[2*i]).crop(crop_box)
        image_00.save(dis_folder + all_images[2*i][66:])
        image_1 = Image.open(all_images[2*i+1]).crop(crop_box)
        image_1.save(dis_folder + all_images[2*i+1][66:])

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

    for i in range(len(all_images)/2):
        if i%1000==0:
            print('Writing %s.tfrecord'%(i/1000)) 
            if i!=0:
                writer.close()
            writer = tf.python_io.TFRecordWriter('croppedImage_tfrecord/croppedImage_%s'%(i/1000) + '.tfrecord')
      
        # read cropped image in jpg
        img_00 = np.array(Image.open(all_images[2*i]))
        img_1 = np.array(Image.open(all_images[2*i+1])).tobytes()

        height, width = img_00.shape[0], img_00.shape[1]
        img_00 = img_00.tobytes()

        # save image and graps data into tfrecord files
        example = tf.train.Example(features=tf.train.Features(
            feature={
            'name': _bytes_feature(all_images[2*i][67:90]),
            'img_1': _bytes_feature(img_1),
            'img_00': _bytes_feature(img_00),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'move_1': _floats_feature(eval(data['move_1'][i][1:])),
            'rotate_angle': _floats_feature([data['rotate_angle'][i]]),
            'success': _floats_feature([data['success'][i]]),
            'isDoll': _floats_feature([data['isDoll'][i]])
            }))
        writer.write(example.SerializeToString())
    writer.close()
    return


# select success grasp and save cropped image 00, 1 and 12 in dis_folder: 
def select_success_grasp(src_folder, dis_folder):
    if not os.path.isdir(dis_folder):
        print('dis folder not exist! now creating a new one ...')
        os.mkdir(dis_folder)

    # list all images
    all_images = sorted(glob.glob(src_folder+'/*.jpg'), key=numericalSort)

    for i in range(len(all_images)/2):
        if data['success'][i] == 1:
            shutil.copy2(all_images[2*i], dis_folder)   
            shutil.copy2(all_images[2*i+1], dis_folder)         
            
            # determine the location of crop box
            coord = eval(data['move_1'][i][1:])
            angle = data['rotate_angle'][i]
            x = coord[0]
            y = coord[1]
            lftpix = int(1067-615*(y+0.25)/0.6)-50
            uppix = int(810-480*(x+0.73)/0.46)-50
            shft=180
            crop_box = (lftpix-shft,uppix-shft,lftpix+shft,uppix+shft)

            image_path = origin_folder + all_images[2*i].split('_')[1][3:] + '/I_' + all_images[2*i].split('_')[3] + '_12_color_camB.jpg'
            image_12 = Image.open(image_path).crop(crop_box)
            image_12.save(dis_folder + '/' + image_path[54:74] + '_' + image_path[75:])
    return

def traverse_theta(src_folder, dis_folder):
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
    all_images = sorted(glob.glob(src_folder+'/*_00_color*.jpg'), key=numericalSort)

    for i in range(len(all_images)):
        if i%1000==0:
            print('Writing %s.tfrecord'%(i/1000))
            if i!=0:
                writer.close()
            writer = tf.python_io.TFRecordWriter('croppedImage_traversed_tfrecord/croppedImage_%s'%(i/1000) + '.tfrecord')

        # read cropped image in jpg
        img_00 = np.array(Image.open(all_images[i]).resize((227, 227), Image.ANTIALIAS)).tobytes()

        # traverse over indicator j
        # label {0:fail, 1:success, 2:unknown, 3:conflict}
        labels = []
        theta_idx = (data['rotate_angle'][i]+3.14)//(2*2*3.14/NUM_THETAS) + 1
        for j in range(NUM_THETAS+1):
            if data['isDoll'][i]==0 and j==0:
                labels.append(0)
                continue
            if data['isDoll'][i]==0 and j>0:
                labels.append(3)
                continue
            if data['isDoll'][i]==1 and j==0:
                labels.append(3)
                continue
            if theta_idx==j:
                labels.append(data['success'][i])
                continue
            labels.append(2)
        labels = np.array(labels)

        # save image and graps data into tfrecord files
        example = tf.train.Example(features=tf.train.Features(
            feature={
            'img_00': _bytes_feature(img_00),
            'rotate_angle': _floats_feature([data['rotate_angle'][i]]),
            'success': _floats_feature([data['success'][i]]),
            'isDoll': _floats_feature([data['isDoll'][i]]),
            'label': _floats_feature(labels)
            }))
        writer.write(example.SerializeToString())
    writer.close()
    return

# save the data in src_folder into tfrecord in dis_folder
#crop_image(rename_origin_folder, './croppedImage_jpg')
#tf_writer('./croppedImage_jpg', './croppedImage_tfrecord')
traverse_theta('./croppedImage_jpg', './croppedImage_traversed_tfrecord')
#select_success_grasp('./croppedImage_jpg', './croppedImage_jpg_success')
