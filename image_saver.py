import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2, os




class PositionTracker:
	def __init__(self):
		rospy.init_node('kinect_tracker')
		self.sub = rospy.Subscriber('kinect2/qhd/image_color', Image, self.image_callback, queue_size=1)
		self.bridge = CvBridge()
		


	def image_callback(self, data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
		except CvBridgeError as e:
			print(e)
		#cv2.imshow('Image window', cv_image)
		result = cv2.imwrite('new.png', cv_image)
		#cv2.waitKey(3)
		self.sub.unregister()
		rospy.signal_shutdown('reason')






def image_save():
	ic = PositionTracker()	
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print('shut down')
	return True


image_save()


'''
def image_callback(msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        cv2.imwrite('camera_image.jpeg', cv2_img)	
        sub_once.unregister()
	rospy.signal_shutdown('reason')
	print 'test'       



# Instantiate CvBridge
bridge = CvBridge()
sub_once = None
rospy.init_node('image_listener')
# Define your image topic
image_topic = "kinect2/qhd/image_color"
# Set up your subscriber and define its callback
sub_once = rospy.Subscriber(image_topic, Image, image_callback)
# Spin until ctrl + c
rospy.spin()
'''


