import os, sys
from compare_images import *
from cut_img import *
import shutil
import csv
import pandas
import numpy as np


def get_lebel(data_dir):
    #target data dir


    #make the temporary dir to store the cut img
    os.chdir(data_dir)
    output_dir = data_dir + '/cut_img/'
    #output_dir = os.path.join(data_dir, 'cut_img')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    #cut img
    for img_name in os.listdir(data_dir):
        if img_name.split('.')[-1] == 'jpg':
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
    try:
        for i in range(0, 40):
            sim = 100
            # in first img compare the I_n_00 and I_n+1_00
            if i == 0:
                imga = output_dir + 'I_1_00_color_camA_cut.jpg'
                imgb = output_dir + 'I_2_00_color_camA_cut.jpg'
                sim = compare_image(imga, imgb)
                #print 'first sim:' + str(sim)
            # other img compare the I_n_13 and I_n+1_13
            # I_n_13 is more accurate than other img
            else:
                imga = output_dir + 'I_' + str(i) + '_13_color_camA_cut.jpg'
                imgb = output_dir + 'I_' + str(i + 1) + '_13_color_camA_cut.jpg'
                sim = compare_image(imga, imgb)
                #print i+1
                #print sim
            if sim < 55 and gSTA_list[i] != '3':
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
    base_dir = '/home/ancora-sirlab/arcade_claw_test/test'

    for dirs in os.listdir(base_dir):
        #print(dirs)
    #data_dir = '/home/ancora-sirlab/arcade_claw_test/2017-04-04 14:06:44/'
        if dirs.split(' ')[0] == '2017-04-25':
            data_dir = os.path.join(base_dir, dirs)
            label = get_lebel(data_dir)
    #print(data_dir)
   # print(label)
            data = pandas.read_csv(os.path.join(data_dir, 'data.csv'))
            labels = np.array([0 for _ in range(len(data['move_00']))])
            labels[label] = 1
            data['success'] = labels
            data.to_csv(os.path.join(data_dir, 'data.csv'), index=False)





