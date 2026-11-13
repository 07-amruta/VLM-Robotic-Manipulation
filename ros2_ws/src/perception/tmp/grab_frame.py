import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

OUT = '/workspaces/VLM-Robotic-Manipulation/frame.png'

rclpy.init()
node = Node('grab_frame')
frames = []
node.create_subscription(Image, '/scene_camera/image', frames.append, 1)
while rclpy.ok() and not frames:
    rclpy.spin_once(node, timeout_sec=0.5)
if frames:
    cv2.imwrite(OUT, CvBridge().imgmsg_to_cv2(frames[0], 'bgr8'))
    print('saved', OUT, frames[0].width, 'x', frames[0].height, frames[0].encoding)
node.destroy_node()
rclpy.shutdown()