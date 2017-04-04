from PIL import Image

import os

def cut_img(img_name, output_dir):
	img = Image.open(img_name)
	lft = 1160
	up = 440
	#shift = 360
	box = (lft, up, lft+500, up+500)

	img = img.crop(box)
	img.save(output_dir + img_name.split('.')[0] + '_cut.jpg')
	return img


if __name__ == '__main__':
	path = './'
	output_dir = path+ 'cut_img/'
	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
	#for img_name in os.listdir(path):
	#	if img_name.split('.')[-1] == 'jpg':
	#		cut_img(img_name, output_dir)
	#img_name = 'cam1_I_1_1_color.jpg'
	img_name = 'I_1_00_color_camA.jpg'
	cut_img(img_name, output_dir)

