### robot_gripper.py

import os
import json
import requests
import base64
import cv2
from datetime import datetime
import time
from grasping.grasping import GripperOperations
from utils.logging_util import setup_logger

# Set debug mode based on environment variable
Debug_1 = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')

# Ensure the image directory exists
image_dir = os.getenv('IMAGE_SAVE_DIR', './images')
os.makedirs(image_dir, exist_ok=True)

class PhysicalRobotGripper:
	
	def __init__(self, robot_name, token, object_api_url):
		if not token or not object_api_url:
			raise ValueError("Missing required environment variables: CLOUDGRIPPER_TOKEN or DETECTION_API_URL.")

		self.token = token
		self.object_api_url = object_api_url
		self.robot = GripperOperations(robot_name, self.token)
		self.robot_name = robot_name
		self.sequence_num = 1
		self.wait_time_move_xy = 2
		self.wait_time_open_rotation = 0.5
		self.station_coordination = [0.0, 0.0]

		try:
			self.logger = setup_logger(f"{__name__}.{self.robot_name}")
		except Exception as e:
			import logging
			self.logger = logging.getLogger(f"{__name__}.{self.robot_name}")
			logging.basicConfig(level=logging.DEBUG if Debug_1 else logging.INFO)
			self.logger.error(f"Logger setup failed: {e}")

		if Debug_1:
			self.logger.info(f"Initialized PhysicalRobotGripper with robot name: {self.robot_name}")
			#self.logger.debug(f"Using token: {self.token}")
			#self.logger.debug(f"Object Detection API URL: {self.object_api_url}")


	def parse_detections(self, detections):
		obj = detections.get("objects", {})
		detection_t = {
			'label': obj.get('label', 'Unknown'),
			'confidence': round(obj.get('confidence', 0.0), 2),
			'x': round(obj.get('x', 0.0), 2),
			'y': round(obj.get('y', 0.0), 2),
			'z': round(obj.get('z', 0.0), 2),
			'width': round(obj.get('width', 0.0), 2),
			'height': round(obj.get('height', 0.0), 2),
			'rotation': obj.get('rotation', 90) }

		if Debug_1: self.logger.info(f"  The reqested target (conf:{detection_t['confidence']}) with x:{detection_t['x']} and y:{detection_t['y']}")

		return detection_t

	def sending_image_and_getting_info_via_comm(self, image, image_file_name=None, mode="camera"):
		try:
			if mode == "file":
				with open(image, 'rb') as image_file:
					encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
			elif mode == "camera":
				success, buffer = cv2.imencode('.jpg', image)
				if not success:
					self.logger.error("Failed to encode image.")
					return None
				encoded_image = base64.b64encode(buffer).decode('utf-8')
			else:
				self.logger.error(f"Invalid mode: {mode}")
				return None

			headers = {
				'Content-Type': 'application/json',
				'Authorization': f'Bearer {self.token}'
			}

			payload = json.dumps( {
				'actor_id': self.robot_name,
				'label_id': 0,
				'image_data': encoded_image,
				'image_file_name': image_file_name,
				'sequence_num': self.sequence_num
			})
			self.logger.info(f">>> #{self.sequence_num}-th request is transmited")
			self.sequence_num = self.sequence_num + 1

			response = requests.post(self.object_api_url, headers=headers, data=payload, timeout=120)
			response.raise_for_status()

			if 'OD-Request-Latency' in response.headers:
				if 'Sequence-Num' in response.headers:
					latency = response.headers['OD-Request-Latency']
					sequence_num_t = response.headers['Sequence-Num']
					self.logger.info(f"<< #{sequence_num_t}-th reply has a latency ({latency})")
				else:
					latency = response.headers['OD-Request-Latency']
					self.logger.info(f"<<< A request has a latency ({latency})")				
			else:
				self.logger.warning("OD-Request-Latency header not found in the response.")

			detections = response.json()
			return self.parse_detections(detections)

		except requests.RequestException as e:
			self.logger.error(f"Request failed: {e}")
		except (ValueError, json.JSONDecodeError) as e:
			self.logger.error(f"Invalid response from detection API: {e}")

		return None

	def log_action(self, action, info, timestamp):
		readable_date = datetime.fromtimestamp(timestamp).strftime('%d_%H_%M_%S_%f') if timestamp else datetime.now().strftime('%d_%H_%M_%S_%f')
		self.logger.info(f"{self.robot_name} - {action} - {info} at {readable_date}")

	def back_station_robot(self):
		try:
			timestamp = self.robot.move_xy(self.station_coordination[0], self.station_coordination[1])
			self.log_action("Returning to station", f"x:{self.station_coordination[0]}, y:{self.station_coordination[1]}", timestamp)
			time.sleep(self.wait_time_move_xy)
		except Exception as e:
			self.logger.error(f"Failed to return to station: {e}")

	def task_robot_action(self, object_info):
		try:
			timestamp = self.robot.move_xy(object_info['x'], object_info['y'])
			self.log_action("Moving to", object_info, timestamp)

			time.sleep(self.wait_time_move_xy)

			timestamp = self.robot.gripper_open()
			self.log_action("Opening gripper", {}, timestamp)

			time.sleep(self.wait_time_open_rotation)

			timestamp = self.robot.rotate(object_info['rotation'])
			self.log_action("Rotating", object_info, timestamp)

			img_base, timestamp = self.robot.getImageBase()
			self.save_image(img_base, timestamp)

			state, timestamp = self.robot.get_state()
			self.log_action("Retrieving state", {'state': state}, timestamp)
		except Exception as e:
			self.logger.error(f"Task failed: {e}")

	def save_image(self, image, timestamp):
		try:
			if image is not None:
				readable_date = datetime.fromtimestamp(timestamp).strftime('%d_%H_%M_%S_%f') if timestamp else datetime.now().strftime('%d_%H_%M_%S_%f')
				image_file_name = f"{image_dir}/{self.robot_name}_bottom_{readable_date}.jpg"
				cv2.imwrite(image_file_name, image)
				self.logger.info(f"Image saved as {image_file_name}")
			else:
				self.logger.warning("No image to save.")
		except Exception as e:
			self.logger.error(f"Failed to save image: {e}")

