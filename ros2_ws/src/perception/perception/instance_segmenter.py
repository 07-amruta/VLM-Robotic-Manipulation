import cv2
import numpy as np
from cv_bridge import CvBridge
from perception_msgs.msg import InstanceMask, InstanceMaskArray
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO

from perception.ros_utils import ConfiguredNode, run_node


class InstanceSegmenter(ConfiguredNode):
    """Closed-set producer: YOLO segmentation + tracking on the colour image."""

    def __init__(self):
        super().__init__('instance_segmenter')
        self.bridge = CvBridge()
        self.info = None
        self.model = YOLO(self.param('yolo_model'))
        latest = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, self.param('color_info_topic'), self.on_info, latest)
        self.create_subscription(Image, self.param('color_topic'), self.on_image, latest)
        self.pub = self.create_publisher(InstanceMaskArray, 'instance_masks/yolo', 10)

    def on_info(self, msg):
        self.info = msg

    def on_image(self, msg):
        if self.info is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        result = self.model.track(
            frame, persist=True, tracker=self.param('yolo_tracker'), retina_masks=True,
            conf=self.param('yolo_conf'), device=self.param('yolo_device'),
            quantize=self.param('yolo_quantize'), verbose=False)[0]

        out = InstanceMaskArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.param('color_frame')
        out.camera_info = self.info
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy() > 0.5
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            ids = None if result.boxes.id is None else result.boxes.id.int().cpu().numpy()
            erode = self.param('mask_erode_px')
            for i, m in enumerate(masks):
                m = m.astype(np.uint8) * 255
                if erode > 0:
                    m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=erode)
                if not m.any():
                    continue
                inst = InstanceMask()
                inst.track_id = int(ids[i]) if ids is not None else -1
                inst.class_name = result.names[classes[i]]
                inst.score = float(scores[i])
                inst.mask = self.bridge.cv2_to_imgmsg(m, 'mono8')
                out.instances.append(inst)
        self.pub.publish(out)


def main():
    run_node(InstanceSegmenter)