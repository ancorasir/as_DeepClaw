# coding: utf-8
# author: Xiaoyi He
import numpy as np
import cv2
import sys
import platform
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

class ImageSaver(object):
	"""a class for image saving"""
	def __init__(self):
		# if run on the Nvidia Jetson, self.platform = 0
		self.platform = 0 if platform.platform().split('-')[2] == 'tegra' else 1
		try:
		    from pylibfreenect2 import OpenCLPacketPipeline
		    self.pipeline0 = OpenCLPacketPipeline()
                    self.pipeline1 = OpenCLPacketPipeline()
                    self.pipeline2 = OpenCLPacketPipeline()
		except:
		    from pylibfreenect2 import CpuPacketPipeline
		    self.pipeline0 = CpuPacketPipeline()
                    self.pipeline1 = CpuPacketPipeline()
                    self.pipeline2 = CpuPacketPipeline()

		self.fn = Freenect2()
		
		num_devices = self.fn.enumerateDevices()
		print(num_devices)
		if num_devices == 0:
		    print("No device connected!")
		    return

		self.serial0 = self.fn.getDeviceSerialNumber(0)
		self.device0 = self.fn.openDevice(self.serial0, pipeline=self.pipeline0)

		self.listener0 = SyncMultiFrameListener(
		    FrameType.Color | FrameType.Ir | FrameType.Depth)
		# Register listeners
		self.device0.setColorFrameListener(self.listener0)
		self.device0.setIrAndDepthFrameListener(self.listener0)

		

		self.serial1 = self.fn.getDeviceSerialNumber(2)
		
		self.device1 = self.fn.openDevice(self.serial1, pipeline=self.pipeline1)
		
		self.listener1 = SyncMultiFrameListener(
		    FrameType.Color | FrameType.Ir | FrameType.Depth)
		# Register listeners
		self.device1.setColorFrameListener(self.listener1)
		self.device1.setIrAndDepthFrameListener(self.listener1)
		

		'''
		self.serial2 = self.fn.getDeviceSerialNumber(1)
		
		self.device2 = self.fn.openDevice(self.serial2, pipeline=self.pipeline2)
		
		self.listener2 = SyncMultiFrameListener(
		    FrameType.Color | FrameType.Ir | FrameType.Depth)
		# Register listeners
		self.device2.setColorFrameListener(self.listener2)
		self.device2.setIrAndDepthFrameListener(self.listener2)

		'''

		self.device0.start()
		self.cap0 = cv2.VideoCapture(self.serial0)
		self.device1.start()
		self.cap1 = cv2.VideoCapture(self.serial1)
		

	def __del__(self):
		self.device0.stop()
		self.device0.close()
		self.cap0.release()

		self.device1.stop()
		self.device1.close()
		self.cap1.release()

		self.device2.stop()
		self.device2.close()
		self.cap2.release()

		

	def kinect_close(self):
		self.device0.stop()
		self.device0.close()

		self.device1.stop()
		self.device1.close()

	def lifcam_close(self):
		# When everything done, release the capture
		self.cap.release()
		self.cap1.release()
	def kinect_saver(self, prefix):
		'''
		Save the image and depth data captured by kinect camera.
		Image saves in .jpg fomat, depth data saved as numpy array.

		Args:
			prefix: the prefix of the save files.
		'''
		
		print('start photoing')
		
		frames0 = None
		frames0 = self.listener0.waitForNewFrame()
		
		
		#print(1)
		color = frames0["color"]
		#print(2)
		depth = frames0["depth"]
		#print(3)
		color = cv2.resize(color.asarray(),
		                           (int(1920), int(1080)))
		#print(4)
		color = cv2.flip(color, 1)
		#print(5)
		# rgb channels are in reverse order in Nvidia Jetson
		if self.platform == 0:
			(r, g, b, d) = cv2.split(color)
			color = cv2.merge([b, g, r])
		# save image in qhd size
		cv2.imwrite(prefix + '_color_camB.jpg', color)
		#print(6)
		# cv2.imwrite(prehttps://github.com/puzzledqs/BBox-Label-Tool.gitfix + '_depth.jpg', cv2.flip(depth.asarray(), 1))
		np.save(prefix + '_depth_camB.npy', cv2.flip(depth.asarray(), 1))
		#print(7)
		self.listener0.release(frames0)
		print('save B')
		
		

		
		
		frames1 = None
		frames1 = self.listener1.waitForNewFrame()
		color = frames1["color"]
		depth = frames1["depth"]
		color = cv2.resize(color.asarray(),
		                           (int(1920), int(1080)))
		color = cv2.flip(color, 1)
		# rgb channels are in reverse order in Nvidia Jetson
		if self.platform == 0:
			(r, g, b, d) = cv2.split(color)
			color = cv2.merge([b, g, r])
		# save image in qhd size
		cv2.imwrite(prefix + '_color_camC.jpg', color)
		# cv2.imwrite(prefix + '_depth.jpg', cv2.flip(depth.asarray(), 1))
		np.save(prefix + '_depth_camC.npy', cv2.flip(depth.asarray(), 1))
		self.listener1.release(frames1)

		print('save C')
		

		self.device2.start()
		self.cap2 = cv2.VideoCapture(self.serial2)
		frames2 = None
		frames2 = self.listener2.waitForNewFrame()
		color = frames2["color"]
		depth = frames2["depth"]
		color = cv2.resize(color.asarray(),
		                           (int(1920), int(1080)))
		color = cv2.flip(color, 1)
		# rgb channels are in reverse order in Nvidia Jetson
		if self.platform == 0:
			(r, g, b, d) = cv2.split(color)
			color = cv2.merge([b, g, r])
		# save image in qhd size
		cv2.imwrite(prefix + '_color_camA.jpg', color)
		# cv2.imwrite(prefix + '_depth.jpg', cv2.flip(depth.asarray(), 1))
		np.save(prefix + '_depth_camA.npy', cv2.flip(depth.asarray(), 1))
		self.listener2.release(frames2)

		print('save A')
		self.device2.stop()
		
		

	def lifcam_saver(self, file_name):
		'''
		Save the image captured by Webcam.

		Args:
			file_name: the file name of the saving image.
		'''
		# Capture frame-by-frame
		ret, frame = self.cap.read()
		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
		# Display the resulting frame
		cv2.imwrite(file_name, gray)
		
		
