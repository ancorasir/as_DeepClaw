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

        except:
            from pylibfreenect2 import CpuPacketPipeline
            self.pipeline0 = CpuPacketPipeline()
            self.pipeline1 = CpuPacketPipeline()

        self.fn = Freenect2()

        #get the available devices
        num_devices = self.fn.enumerateDevices()
        print(num_devices)
        if num_devices == 0:
            print("No device connected!")
            return

        #open device0, Camera B
        self.serial0 = self.fn.getDeviceSerialNumber(0)
        self.device0 = self.fn.openDevice(self.serial0, pipeline=self.pipeline0)

        #open new listener for shot image
        self.listener0 = SyncMultiFrameListener(
            FrameType.Color | FrameType.Ir | FrameType.Depth)

        # Register listeners, color and depth
        self.device0.setColorFrameListener(self.listener0)
        self.device0.setIrAndDepthFrameListener(self.listener0)

        #open device1 Camera A
        self.serial1 = self.fn.getDeviceSerialNumber(1)
        self.device1 = self.fn.openDevice(self.serial1, pipeline=self.pipeline1)

        #open new listener for shot image
        self.listener1 = SyncMultiFrameListener(
            FrameType.Color | FrameType.Ir | FrameType.Depth)

        # Register listeners, color and depth
        self.device1.setColorFrameListener(self.listener1)
        self.device1.setIrAndDepthFrameListener(self.listener1)

        #start device0
        self.device0.start()
        self.cap0 = cv2.VideoCapture(self.serial0)

        #start device1
        self.device1.start()
        self.cap1 = cv2.VideoCapture(self.serial1)

    #close all
    def __del__(self):
        self.device0.stop()
        self.device0.close()
        self.cap0.release()

        self.device1.stop()
        self.device1.close()
        self.cap1.release()

    #not use
    def kinect_close(self):
        self.device0.stop()
        self.device0.close()

        self.device1.stop()
        self.device1.close()

    #not use
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

        #device0
        #get new shot frame
        frames0 = self.listener0.waitForNewFrame()

        #get color data
        color = frames0["color"]

        #get depth data
        depth = frames0["depth"]

        #resize
        color = cv2.resize(color.asarray(),
                           (int(1920), int(1080)))


        color = cv2.flip(color, 1)


        if self.platform == 0:
            (r, g, b, d) = cv2.split(color)
            color = cv2.merge([b, g, r])

        #save color img
        cv2.imwrite(prefix + '_color_camB.jpg', color)
        #save depth img
        np.save(prefix + '_depth_camB.npy', cv2.flip(depth.asarray(), 1))

        #release listener for next shot
        self.listener0.release(frames0)
        print('save B')


        #device1, same as device0
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
        cv2.imwrite(prefix + '_color_camA.jpg', color)
        np.save(prefix + '_depth_camA.npy', cv2.flip(depth.asarray(), 1))
        self.listener1.release(frames1)

        print('save A')

    #not use
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
