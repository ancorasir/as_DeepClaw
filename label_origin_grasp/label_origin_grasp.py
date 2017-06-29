import os, sys
import shutil
import csv
import pandas
import numpy as np

from compare_images import *
from cut_img import *


def get_lebel(data_dir):
    #target data dir


    #make the temporary dir to store the cut img
    os.chdir(data_dir)
    output_dir = data_dir + '/cut_img/'
    #output_dir = os.path.join(data_dir, 'cut_img')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    #record the total grasp number in this grasp circle
    total_grasp_num = 0

    #cut img
    for img_name in os.listdir(data_dir):
        if img_name.split('.')[-1] == 'jpg':
            #get the image information from image name
            temp_img_name_list = img_name.split('.')[0].split('_')
            type = temp_img_name_list[-1]
            iter = temp_img_name_list[2]
            total_grasp_num = max(total_grasp_num, int(temp_img_name_list[1]))
            if type == 'camB' and (iter == '00' or iter == '12'):
                cut_img(img_name, output_dir)

    #save the gSTA status of gripper in 'data.csv'
    #gSTA = 3 means gripper has fully closed, which means it has no grasp anything
    gSTA_list = []

    # read csv file and gSTA
    with open(os.path.join(data_dir, 'data.csv'), 'rb') as f:
        reader = csv.DictReader(f)
        # print reader
        for row in reader:
            # print row
            gSTA = row['gSTA']
            gSTA_list.append(gSTA)

    #print gSTA_list



    #compare img and gSTA status
    grasp_num = 0
    label = []
    threshold = 54
    print 'total grasp num in this grasp circle:'
    print total_grasp_num
    try:
        #for i in range(1, total_grasp_num - 1): modifify 31/05/2017
        for i in range(1, total_grasp_num):
            sim = 100
            imga = output_dir + 'I_' + str(i) + '_12_color_camB_cut.jpg'
            imgb = output_dir + 'I_' + str(i + 1) + '_00_color_camB_cut.jpg'
            sim = compare_image(imga, imgb)
            #print i, sim
            if sim < threshold and gSTA_list[i-1] != '3':
                #print i + 1
                label.append(i)
                grasp_num += 1
    except IOError:
        print('image not exist!')


    #print successfully grasp number
    print 'totol success number:'
    print grasp_num

    #delete the temporary cut img dir
    os.chdir(data_dir)
    shutil.rmtree(output_dir)

    return label

if __name__ == '__main__':
    base_dir = '/home/ancora-sirlab/arcade_claw_test/origin_grasp_data'

    for dirs in os.listdir(base_dir):
        if dirs.split(' ')[0].split('-')[0] == '2017':
        #if dirs == '2017-05-07 11:52:37':
            data_dir = os.path.join(base_dir, dirs)
            print(data_dir)
            label = get_lebel(data_dir)

            print(label)
            data = pandas.read_csv(os.path.join(data_dir, 'data.csv'))
            labels = np.array([0 for _ in range(len(data['move_00']))])
            #labels[label] = 1 modifify 31/05/2017
            labels[np.array(label)-1] = 1
            data['success'] = labels
            data.to_csv(os.path.join(data_dir, 'data.csv'), index=False)
