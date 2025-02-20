### object_detection_cloud.py
# services/object_detection_cloud.py

import os
import cv2
import base64
import logging
import requests
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from utils.logging_util import setup_logger

# If you want debug logs from this code, set DEBUG=True in your environment
Debug = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')

class CloudModelConnector:
	"""
	A "cloud" manager that sends all object detection requests to a single
	aggregator endpoint, e.g., http://129.192.82.37:30809/detect_cloud.
	"""

	def __init__(self, num_workers=3):
		"""
		:param num_workers: Number of threads to use for asynchronous calls.
		"""
		# Setup logger
		try:
			self.logger = setup_logger(f"{__name__}.CloudModelConnector")
		except Exception as e:
			self.logger = logging.getLogger(f"{__name__}.CloudModelConnector")
			logging.basicConfig(level=logging.DEBUG if Debug else logging.INFO)
			self.logger.error(f"Logger setup failed: {e}")

		self.endpoint_url = os.getenv('CLOUD_DETECTION_API_URL')
		self.executor = ThreadPoolExecutor(max_workers=num_workers)
		self.logger.info(f"[CloudModelConnector] Sending detections to {self.endpoint_url}")

	def cloud_detect_objects_async(self, image, model_quality, actor_id, sequence_num):
		"""
		Submits an asynchronous job to the thread pool that sends the given
		image (and requested model_quality) to the cloud aggregator.

		:param image: (np.ndarray) OpenCV BGR image array.
		:param model_quality: (str) "low", "medium", or "high".
		:return: A Future object. You can call future.result() to get the detections.
		"""
		future = self.executor.submit(self._cloud_detect_objects, image, model_quality, actor_id, sequence_num)
		return future

	def _cloud_detect_objects(self, image, model_quality, actor_id, sequence_num):
		"""
		Worker method that actually performs the network request to the cloud endpoint.

		:param image: (np.ndarray) OpenCV BGR image array.
		:param model_quality: (str) "low", "medium", or "high".
		:return: A list of detection dicts, each containing keys like:
			{
			   "label": <int>,
			   "confidence": <float>,
			   "bbox": (x_norm, y_norm, w_norm, h_norm)
			}
			If the cloud service returns only a single "best object," we adapt it to
			a list with one element.
		"""
		# 1) Encode image as JPEG
		success, encoded_image = cv2.imencode(".jpg", image)
		if not success or encoded_image is None:
			raise ValueError("Failed to encode image for sending to cloud aggregator.")

		# 2) Convert to base64
		b64_data = base64.b64encode(encoded_image).decode('utf-8')

		# 3) Prepare JSON payload
		# Adjust keys as needed based on how your cloud aggregator expects them.
		payload = json.dumps( {
			'actor_id': actor_id,
			'sequence_num':	sequence_num,
			'image_data': b64_data,
			'model_quality': model_quality
		})

		# 4) Send POST request to the aggregator
		try:
			if Debug:
				self.logger.debug(f">>>>>>[CloudModelConnector], ID({g.actor_id})-SN({g.sequence_num}), is posting to {self.endpoint_url} "
							  f"with model_quality={model_quality}")
			response = requests.post(self.endpoint_url, data=payload, timeout=30)
			response.raise_for_status()
		except requests.RequestException as e:
			self.logger.error(f"[CloudModelConnector] Request to {self.endpoint_url} failed: {e}")
			raise

		# 5) Parse JSON result
		result_json = response.json()
		if "detections" not in result_json:
			self.logger.error(f"Cloud service response missing 'detections' key: {result_json}")
			raise ValueError("Invalid response format from microservice.")

		detections = result_json["detections"]
		self.logger.info(f"[CloudModelConnector] Received {len(detections)} detections.")

		# Return a dictionary with additional information
		return {
			"actor_id": actor_id,
			"sequence_num": sequence_num,
			"detections": detections
		}

	def shutdown(self):
		"""
		Gracefully shuts down the thread pool executor.
		"""
		self.logger.info("[CloudModelConnector] Shutting down thread pool.")
		self.executor.shutdown(wait=True)
