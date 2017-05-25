from PIL import Image
import os, sys
import shutil
import csv
import pandas as pd
import numpy as np

def cut_img(img_name,output_dir,i,data_dir):
	img = Image.open(data_dir+img_name)
	data = pd.read_csv(data_dir+'/data.csv')
	# get the coordinate of each grasp from data.csv
	# skip the 'p[' at the front of move_1
	coord = eval(data['move_1'][i-1][1:])
        x = coord[0]
        y = coord[1]
	#print(i+1)
        #print('x:%f' %x)
	#print('y:%f' %y)

	#mapping the coordinates in robotic system and photo taken by camera B
        lftpix = int(1067-615*(y+0.25)/0.6)-50
        uppix = int(810-480*(x+0.73)/0.46)-50
	#lft = 500
	#up = 330
	#rigt = 1050
        #down= 810

	# Size of new images are (2*shft,2*shft) 
	shft=180
	box = (lftpix-shft,uppix-shft,lftpix+shft,uppix+shft)

	img = img.crop(box)
	
	img.save(output_dir +'/' + img_name.split('.')[0] + '_cut.jpg')
	return img

def cut_perfolder(data_dir,dirs):
   
    os.chdir(data_dir)
    output_dir = base_dir + '/cut_img/' + dirs
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    #record the total grasp number in this grasp circle
    total_grasp_num = 0
    
    files_dir=data_dir+'/data.csv'
    shutil.copy(files_dir,output_dir)
    #cut img
    for img_name in os.listdir(data_dir):
        if img_name.split('.')[-1] == 'jpg':
            #get the image information from image name
            temp_img_name_list = img_name.split('.')[0].split('_')
            type = temp_img_name_list[-1]
            iter = temp_img_name_list[2]
            total_grasp_num = max(total_grasp_num, int(temp_img_name_list[1]))
 
    for i in range(1, total_grasp_num + 1):
        imga = '/I_' + str(i) + '_00_color_camB.jpg'
        imgb = '/I_' + str(i) + '_1_color_camB.jpg'
        cut_img(imga, output_dir,i,data_dir)
        cut_img(imgb, output_dir,i,data_dir)


		    
    return True

if __name__ == '__main__':
    base_dir = '/home/ancora-sirlab/arcade_claw_test/origin_grasp_data'

    for dirs in os.listdir(base_dir):
        if dirs.split(' ')[0].split('-')[0] == '2017'and  dirs.split(' ')[0].split('-')[1] == '05' and dirs.split(' ')[0].split('-')[2] == '09' and dirs.split(' ')[1].split(':')[0] == '19':
            data_dir = os.path.join(base_dir, dirs)
            print(data_dir,dirs)
            cut_perfolder(data_dir,dirs)

