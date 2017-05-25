""" 
     cut the images into smaller size ones and the centre of the returned images is exactly the grasping point 
"""

from PIL import Image
import csv
import os
import pandas as pd
import shutil

def cut_img(data_dir,output_dir,i):
	img = Image.open(data_dir+'/I_'+str(i+1)+'_00_color_camB.jpg')
	img2 = Image.open(data_dir+'/I_'+str(i+1)+'_1_color_camB.jpg')
	data = pd.read_csv(data_dir+'/data.csv')
	# get the coordinate of each grasp from data.csv
	# skip the 'p[' at the front of move_1
	coord = eval(data['move_1'][i][1:])
        angle = data['rotate_angle'][i]
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
	img2 = img2.crop(box)

	img.save(output_dir + '/I_'+str(i+1)+'_00_color_camB_cut.jpg')
	img2.save(output_dir + '/I_'+str(i+1)+'_1_color_camB_cut.jpg')
	return img

def cut_perfolder(data_dir,dirs,base_dir):
	#cut images per folder
	output_dir=base_dir+ '/cut_image'
	output_dir=os.path.join(output_dir,dirs)
	if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
	#copy the data.csv in the /arcade_claw_test to the new folder
	files_dir=data_dir+'/data.csv'
	shutil.copy(files_dir,output_dir)
 		
	for i in range(0,40): #number of photos 
            cut_img(data_dir,output_dir,i)
	
	return True

if __name__ == '__main__':
	base_dir = '/home/ancora-sirlab/arcade_claw_test/test'
        
        for dirs in os.listdir(base_dir):
	    if dirs.split(' ')[0] == '2017-05-06':
		data_dir = os.path.join(base_dir,dirs)
		cut_perfolder(data_dir,dirs,base_dir)


	

