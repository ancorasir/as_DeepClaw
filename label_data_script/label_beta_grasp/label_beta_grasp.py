import os, sys
import shutil
import csv
import pandas
import numpy as np

from skimage.measure import compare_ssim
import cv2
from cut_img import *


def get_lebel(data_dir):
    #target data dir
    data_dir_B = data_dir+'/'+data_dir.split('-')[2]+'-'+data_dir.split('-')[3]+'-ImgColorCamB'

    #make the temporary dir to store the cut img
    os.chdir(data_dir_B)
    output_dir = data_dir_B + '/cut_img/'
    #output_dir = os.path.join(data_dir, 'cut_img')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    #record the total grasp number in this grasp circle
    total_grasp_num = 0

    #cut img in data_dir/*-ImgColorCamB/
    for img_name in os.listdir(data_dir_B):
        if img_name.split('.')[-1] == 'jpg':
            #get the image information from image name
            temp_img_name_list = img_name.split('.')[0].split('_')
            type = temp_img_name_list[-1]
            iter = temp_img_name_list[2]
            total_grasp_num = max(total_grasp_num, int(temp_img_name_list[1]))
            if type == 'camB' and (iter == '00' or iter == '12'):
                cut_img(img_name, output_dir)

    #compare img and gSTA status
    grasp_num = 1
    label = []
    threshold = 0.96
    print 'total grasp num in this grasp circle:'
    print total_grasp_num
    try:
        #for i in range(1, total_grasp_num - 1): modifify 31/05/2017
        for i in range(1, total_grasp_num):
            imga = output_dir + 'I_' + str(i) + '_12_color_camB_cut.jpg'
            imgb = output_dir + 'I_' + str(i + 1) + '_00_color_camB_cut.jpg'
            sim = compare_ssim(cv2.imread(imga), cv2.imread(imgb), multichannel=True, full=False)
            if sim < threshold:
                label.append(i)
                grasp_num += 1
    except IOError:
        print('image not exist!')


    #print successfully grasp number
    print 'totol success number:'
    print grasp_num

    #delete the temporary cut img dir
    os.chdir(data_dir_B)
    shutil.rmtree(output_dir)

    return label

if __name__ == '__main__':
    base_dir = '/home/ancora-sirlab/as_DeepClaw_data/beta_grasp_data'

    for dirs in os.listdir(base_dir):
        if dirs.split('-')[0] == 'backuped' and dirs.split('-')[3] == 'ImgData':
       # if dirs == 'backuped-170707-122608-ImgData':
            data_dir = os.path.join(base_dir, dirs)
            print(data_dir)
            label = get_lebel(data_dir)

            print(label)
            data = pandas.read_csv(data_dir+'/'+dirs.split('-')[1]+'-'+dirs.split('-')[2]+ '-data.csv')
            labels = np.array([0 for _ in range(len(data['move_00']))])
            #labels[label] = 1 modifify 31/05/2017
            labels[np.array(label)-1] = 1
            data['success'] = labels
            data.to_csv(data_dir+'/'+dirs.split('-')[1]+'-'+dirs.split('-')[2]+ '-data_labeled.csv', index=False)
