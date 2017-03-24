# coding: utf-8
import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

class ImageSaver(object):
	"""docstring for ImageSaver"""
	def __init__(self):
		try:
		    from pylibfreenect2 import OpenCLPacketPipeline
		    self.pipeline = OpenCLPacketPipeline()
		except:
		    from pylibfreenect2 import CpuPacketPipeline
		    self.pipeline = CpuPacketPipeline()

		self.fn = Freenect2()
		num_devices = self.fn.enumerateDevices()
		if num_devices == 0:
		    print("No device connected!")
		    return

		self.serial = self.fn.getDeviceSerialNumber(0)
		self.device = self.fn.openDevice(self.serial, pipeline=self.pipeline)

		self.listener = SyncMultiFrameListener(
		    FrameType.Color | FrameType.Ir | FrameType.Depth)
		# Register listeners
		self.device.setColorFrameListener(self.listener)
		self.device.setIrAndDepthFrameListener(self.listener)

		self.device.start()
		self.cap = cv2.VideoCapture(0)

	def __del__(self):
		self.device.stop()
		self.device.close()
		self.cap.release()

	def kinect_close(self):
		self.device.stop()
		self.device.close()

	def lifcam_close(self):
		# When everything done, release the capture
		self.cap.release()

	def kinect_saver(self, prefix):
		frames = self.listener.waitForNewFrame()
		color = frames["color"]
		depth = frames["depth"]
		color = cv2.resize(color.asarray(),
		                           (int(1920 / 2), int(1080 / 2)))
		color = cv2.flip(color, 1)
		# save image in qhd size
		cv2.imwrite(prefix + '_color.jpg', color)
		cv2.imwrite(prefix + '_depth.jpg', depth.asarray())
		self.listener.release(frames)

	def lifcam_saver(self, file_name):
		# Capture frame-by-frame
		ret, frame = self.cap.read()
		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
		# Display the resulting frame
		cv2.imwrite(file_name, gray)
		
		
