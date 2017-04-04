import socket

import os, sys
import csv
import time
from image_saver import ImageSaver

#num of batch
#batch_num = 0

#num of grasp attempt per grasp batch
#batch_size = 40

#port and ip address
ip_port = ('192.168.0.108',8892)

#maximum connection number
connect_num = 100

#data folder
#data_path = './new_data/'
data_path = './' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '/'
os.mkdir(data_path)
print '********************************************'
print 'data path: ' + data_path
print '********************************************'

#receive for every socket communication
#blind grasp data collection
def receive_from_robot(conn, iteration, confirm, fcsv, headers, img_saver):

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

	#shot 00 (not use, same as shot 13) 
	if confirm == 'shot_00':			
		img_saver.kinect_saver(data_path + 'I_' + str(iteration) + '_00')
		#img_saver.lifcam_saver('cam2_I_' + str(iteration) + '_00.png')
		conn.send(bytes(confirm))
		print confirm



	
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
		img_saver.kinect_saver(data_path + 'I_' + str(iteration) + '_1')
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
		

		#gripper data
		conn.send(bytes('gripper'))
		gACT = conn.recv(100).split()[2]
		gMOD = conn.recv(100).split()[2]
		gGTO = conn.recv(100).split()[2]
		gSTA = conn.recv(100).split()[2]
		gIMC = conn.recv(100).split()[2]
		gFLT = conn.recv(100).split()[2]
		gPRE = conn.recv(100).split()[2]
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
		
		for i in range(6):
			conn,addr = sock.accept()
			confirm = conn.recv(1024)
			receive_from_robot(conn,iteration, confirm, fcsv, headers, img_saver)


		

#main function
def main(args):

	#connect
	connect_robot(ip_port, connect_num)


if __name__ == '__main__':
	main(sys.argv)




