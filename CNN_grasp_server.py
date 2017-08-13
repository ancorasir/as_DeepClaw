import socket

import os, sys
import csv
import time
from image_saver import ImageSaver
from predictor import Predictor
from PIL import Image

#num of batch
#batch_num = 0

#num of grasp attempt per grasp batch
#batch_size = 40

#port and ip address
ip_port = ('192.168.0.103',8892)

#maximum connection number
connect_num = 100

#data folder
#data_path = './new_data/'
data_path = './' + time.strftime('%Y%m%d-%H%M%S-ImgData',time.localtime(time.time()))[2:] + '/'
model_path = '/home/ancora-sirlab/wanfang/training_cropped_image/checkpoint_100index_alpha'
os.mkdir(data_path)

os.mkdir(data_path + data_path.split('/')[1][:14] + 'ImgColorCamA/')
os.mkdir(data_path + data_path.split('/')[1][:14] + 'ImgDepthCamA/')
os.mkdir(data_path + data_path.split('/')[1][:14] + 'ImgColorCamB/')
os.mkdir(data_path + data_path.split('/')[1][:14] + 'ImgDepthCamB/')

#os.mkdir(data_path + 'ImgColorCamA/')
#os.mkdir(data_path + 'ImgDepthCamA/')
#os.mkdir(data_path + 'ImgColorCamB/')
#os.mkdir(data_path + 'ImgDepthCamB/')

print '********************************************'
print 'data path: ' + data_path
print '********************************************'

#receive for every socket communication
#blind grasp data collection
def receive_from_robot(conn, iteration, confirm, fcsv, headers, img_saver, CNN):

	'''
	args:
		conn:		socket
		iteration:	num of grasp attempt in this batch
		confirm: 	message type from robot controller
		fcsv:		output data csv file		
		headers: 	csv file headers
		ima_saver:	image save class, include two cameras

	return: none
	'''



	'''
	grasp stracture:
	
	***first random move	ITF-SMS0
	shot 00 	no gripper
	move 00		first random move
	shot 01		shot first random move

	***second random move	ITF-MoSo
	move 1		second random move
	shot 1		shot second random move

	***grasp and drop	ITF-Pick
	pick
	shot 11		shot the pick
	move 		to drop box
	shot 12		shot the up position
	drop		drop the toy
	shot 13 	shot the binose the gri

	'''
	print ''
	print '***************! '+str(iteration)+' !***************'

	#shot 00 (used) 
	if confirm == 'shot_00':			
		img_saver.kinect_saver(data_path + 'I_' + str(iteration) + '_00')
		#img_saver.lifcam_saver('cam2_I_' + str(iteration) + '_00.png')
		conn.send(bytes(confirm))
		print 'complete ' + confirm
		print '**********************************************'



	
	#shot 01 (used)	
	elif confirm == 'shot_01':
		print '***ITF-SMS0***********************************'		
		img_saver.kinect_saver(data_path + 'I_' + str(iteration) + '_01')
		#img_saver.lifcam_saver('cam2_I_' + str(iteration) + '_01.png')
		conn.send(bytes(confirm))
		print 'complete ' + confirm
		print '**********************************************'
	

	
	#shot 1	(uesd)	
	elif confirm == 'shot_1':
		print '***ITF-MoSo***********************************'		
		img_saver.kinect_saver(data_path + 'I_' + str(iteration) + '_10')
		#img_saver.lifcam_saver('cam2_I_' + str(iteration) + '_1.png')
		conn.send(bytes(confirm))
		print 'complete ' + confirm
		print '**********************************************'


	
	#shot 11 (used)	
	elif confirm == 'shot_11':	
		print '***ITF-Pick***********************************'	
		img_saver.kinect_saver(data_path + 'I_' + str(iteration) + '_11')
		#img_saver.lifcam_saver('cam2_I_' + str(iteration) + '_11.png')
		conn.send(bytes(confirm))
		print 'complete ' + confirm
		print '**********************************************'


	
	#shot 12 (used)	
	elif confirm == 'shot_12':
		img_saver.kinect_saver(data_path + 'I_' + str(iteration) + '_12')
		#img_saver.lifcam_saver('cam2_I_' + str(iteration) + '_12.png')
		conn.send(bytes(confirm))
		print 'complete ' + confirm
		print '**********************************************'
	


	
	#shot 13 (used)	
	elif confirm == 'shot_13':
		img_saver.kinect_saver(data_path + 'I_' + str(iteration) + '_13')
		#img_saver.lifcam_saver('cam2_I_' + str(iteration) + '_13.png')
		conn.send(bytes(confirm))
		print 'complete ' + confirm
		print '**********************************************'

	
	#move 00 (used)
	#move 1	(used):	
	elif confirm == 'data':

		#move 00
		conn.send(bytes('move_00'))
		random_c_0 = conn.recv(1024)

		#move 1
		conn.send(bytes('move_1'))
		random_c_1 = conn.recv(1024)

		#rotate_angle
		conn.send(bytes('rotate_angle'))
		rotate_angle = conn.recv(1024)
		

		gACT = conn.recv(100)
		gMOD = conn.recv(100)
		gGTO = conn.recv(100)
		gSTA = conn.recv(100)
		gIMC = conn.recv(100)
		gFLT = conn.recv(100)
		gPRE = conn.recv(100)
		#gripper data
		conn.send(bytes('gripper'))
		if gACT :
			gACT = gACT.split()[2]
		else: gACT = 'N'
		if gMOD :
			gMOD = gMOD.split()[2]
		else: gMOD = 'N'
		if gGTO:
			gGTO = gGTO.split()[2]
		else: gGTO = 'N'
		if gSTA:
			gSTA = gSTA.split()[2]
		else: gSTA = 'N'
		if gIMC:
			gIMC = gIMC.split()[2]
		else: gIMC = 'N'
		if gFLT:
			gFLT = gFLT.split()[2]
		else: gFLT = 'N'
		if gPRE:
			gPRE = gPRE.split()[2]
		else: gPRE = 'N'
		
		#gACT = conn.recv(100).split()[2]
		#gMOD = conn.recv(100).split()[2]
		#gGTO = conn.recv(100).split()[2]
		#gSTA = conn.recv(100).split()[2]
		#gIMC = conn.recv(100).split()[2]
		#gFLT = conn.recv(100).split()[2]
		#gPRE = conn.recv(100).split()[2]
		print gACT.split()
		print gMOD.split()
		print gGTO.split()
		print gSTA.split()
		print gIMC.split()
		print gFLT.split()
		print gPRE.split()

		#tcp force 
		conn.send(bytes('force'))
		tcp_force = conn.recv(100).split()[0]
	
		#save data as csv file
		rows = [{'move_00': random_c_0, 
			'move_1': random_c_1, 
			'rotate_angle': rotate_angle, 
			'gACT': gACT, 'gMOD': gMOD, 'gGTO':gGTO, 'gSTA':gSTA, 'gIMC': gIMC, 'gFLT': gFLT, 'gPRE': gPRE,
			'tcp_force': tcp_force }]
		with open(data_path + 'data.csv', 'a') as f:
			fcsv = csv.DictWriter(f, headers)
			fcsv.writerows(rows)

		print 'complete ' + confirm
		print '**********************************************'

	# use CNN
	elif confirm == 'CNN':
		print '***ITF-CEM***********************************'		
		conn.send(bytes(confirm))
		print 'complete ' + confirm
		print '**********************************************'
		img_00 = Image.open(data_path + data_path.split('/')[1][:14] + 'ImgColorCamB/' + data_path.split('/')[1][:14] + 'I_' + str(iteration) + '_00_color_camB.jpg').crop((700, 465, 1165, 875))

		position = conn.recv(1024)
		position = eval(position[1:])
		new_position = CNN.eval(img_00, position) #[x, y, theta]
		print(new_position)
		conn.send(bytes(new_position))
		

	#error
	else:
		print(confirm)
		print 'error'
		print '**********************************************'




#connect to robot and communicate with socket
def connect_robot(ip_port, connect_num):

	'''
	args:	
		ip_port:	ip address and port
		connect_num:	maximun connection number 

	return: none
	'''

	#open the socket server, waiting for data	
	sock = socket.socket()
	sock.bind(ip_port)
	sock.listen(connect_num)
	img_saver = ImageSaver()
	# create CNN module
	CNN = Predictor(model_path)
	print 'connecting the robot'
	print 'waiting for data'
	print '******************************************'
	
	#grasp cycle number
	iteration = 0

	#save data in csv file
	# headers of csv 
	headers = ['move_00', 'move_1', 'rotate_angle','gACT', 'gMOD','gGTO', 'gSTA', 'gIMC', 'gFLT', 'gPRE','tcp_force']
	# create csv file
	with open(data_path + 'data.csv', 'w') as f:
		fcsv = csv.DictWriter(f, headers)
		fcsv.writeheader()

	#start to recieve data
	while True:
		iteration += 1
		
		for i in range(5):
			conn,addr = sock.accept()
			confirm = conn.recv(1024)
			receive_from_robot(conn,iteration, confirm, fcsv, headers, img_saver, CNN)


		

#main function
def main(args):

	#connect
	connect_robot(ip_port, connect_num)


if __name__ == '__main__':
	main(sys.argv)




