import socket

import os, sys


def connect_robot(ip_port, connect_num):
	sock = socket.socket()
	sock.bind(ip_port)
	sock.listen(connect_num)
	print 'waiting for data'
	
	iteration = 0

	while True:
		iteration += 1
		
		for i in range(4):
			conn,addr = sock.accept()
			confirm = conn.recv(1024)
			if confirm == 'c_0':
				#if image_save():
					os.system('python image_saver.py')
					conn.send(bytes("good"))
					os.rename('new.png',str(iteration) + '_c_0.png')
					print 'c_0'

			elif confirm == 'c_1':
				#if image_save():
					os.system('python image_saver.py')
					conn.send(bytes('good'))
					os.rename('new.png',str(iteration) + '_c_1.png')
					print 'c_1'

			elif confirm == 'drop':
				#if image_save():
					os.system('python image_saver.py')
					conn.send(bytes('good'))
					os.rename('new.png',str(iteration) + '_drop.png')
					print 'drop'

			elif confirm == 'data':
				#receiving data
				random_c_0 = conn.recv(1024)
				random_c_1 = conn.recv(1024)
				tcp_force = conn.recv(1024)
				print('data')
				#print data
				print 'random_c_0: ' + str(random_c_0)	
				print 'random_c_1: ' + str(random_c_1)
				print 'tcp_force:' + str(tcp_force)
			else:
				print(confirm)


		


def main(args):
	ip_port = ('192.168.0.108',8892)
	connect_num = 100
	connect_robot(ip_port, connect_num)


if __name__ == '__main__':
	main(sys.argv)
