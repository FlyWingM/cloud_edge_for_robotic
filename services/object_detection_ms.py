# object_detection.py
import os
import cv2
import base64
import logging
import requests
import numpy as np

from utils.logging_util import setup_logger

# If you want debug logs from this code, set DEBUG=True in your environment
Debug_1 = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')

# A simple dictionary mapping model_quality -> microservice endpoint
MICROSERVICE_URLS = {
	"low":    os.getenv('OD_LOW_ENDPOINT',    'http://od-microservice-low:8502/detect'),
	"medium": os.getenv('OD_MEDIUM_ENDPOINT', 'http://od-microservice-medium:8503/detect'),
	"high":   os.getenv('OD_HIGH_ENDPOINT',   'http://od-microservice-high:8504/detect'),
}


class MicroserviceObject:
	"""
	A unified class that picks the correct microservice endpoint based on model_quality
	and sends images to that microservice for object detection. The microservice itself
	loads the TFLite or PyTorch model locally (see microservice_od.py).
	"""

	def __init__(self, model_quality="llow", confidence_threshold=0.1):
		"""
		Initialize the tracker with a selected microservice endpoint.

		:param model_quality: One of ["llow", "low", "medium", "high"] which
							  corresponds to an entry in MICROSERVICE_URLS.
		:param confidence_threshold: A float that will be sent to the microservice
									 to filter detections below this threshold.
		"""
		try:
			self.logger = setup_logger(f"{__name__}.MicroserviceObject")
		except Exception as e:
			self.logger = logging.getLogger(f"{__name__}.MicroserviceObject")
			logging.basicConfig(level=logging.DEBUG if Debug_1 else logging.INFO)
			self.logger.error(f"Logger setup failed: {e}")

		endpoint_url = MICROSERVICE_URLS.get(model_quality.lower())
		if not endpoint_url:
			raise ValueError(
				f"Unknown model quality '{model_quality}'. "
				f"Must be one of {list(MICROSERVICE_URLS.keys())}."
			)
		self.logger.info(f"[MicroserviceObject] Using endpoint '{endpoint_url}' "
						 f"for model quality '{model_quality}'.")
		self.endpoint_url = endpoint_url
		self.confidence_threshold = float(confidence_threshold)

	def detect_objects(self, image):
		"""
		Sends the given image to the configured microservice endpoint for detection.

		:param image: An OpenCV BGR image (H x W x 3) as a NumPy array.
		:return: A list of detections (dicts) from the microservice.
				 Each detection typically has:
					 {
						"label": <int>,
						"confidence": <float>,
						"bbox": (x_norm, y_norm, w_norm, h_norm)
					 }
		"""
		# 1) Encode image as JPEG in memory
		_, encoded_image = cv2.imencode(".jpg", image)
		if encoded_image is None:
			raise ValueError("Failed to encode image for sending to microservice.")

		# 2) Convert to base64
		b64_data = base64.b64encode(encoded_image).decode('utf-8')

		# 3) Prepare JSON payload
		payload = {
			"image_data": b64_data,
			"confidence_threshold": self.confidence_threshold
		}

		# 4) Send POST request to microservice
		try:
			response = requests.post(self.endpoint_url, json=payload, timeout=30)
			response.raise_for_status()
		except requests.RequestException as e:
			self.logger.error(f"Failed to call microservice at {self.endpoint_url}: {e}")
			raise

		# 5) Parse JSON result
		result_json = response.json()
		if "detections" not in result_json:
			self.logger.error(f"Microservice response missing 'detections' key: {result_json}")
			raise ValueError("Invalid response format from microservice.")

		detections = result_json["detections"]
		self.logger.info(f"[MicroserviceObject] Received {len(detections)} detections.")
		return detections

	def debug_detection(self, image, label_id, debug_image_path):
		"""
		Example debug method: calls detect_objects, then draws bounding boxes for
		detections matching `label_id`. Saves or logs a debug image (optional).

		:param image: An OpenCV BGR image (H x W x 3).
		:param label_id: The label/class of interest to visualize.
		:param debug_image_path: Where to save the debug image (optional).
		"""
		all_detections = self.detect_objects(image)
		detections = [d for d in all_detections if d["label"] == label_id]

		original_h, original_w = image.shape[:2]
		for det in detections:
			x_norm, y_norm, w_norm, h_norm = det["bbox"]
			x = int(x_norm * original_w)
			y = int(y_norm * original_h)
			w = int(w_norm * original_w)
			h = int(h_norm * original_h)

			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
			label_text = f"{det['label']}: {det['confidence']:.2f}"
			cv2.putText(image, label_text, (x, y - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# You could save or further analyze the debug image here
		self.logger.info(f"[Debug Detection] Debug image path: {debug_image_path}")
		# Example:
		# cv2.imwrite(debug_image_path, image)
