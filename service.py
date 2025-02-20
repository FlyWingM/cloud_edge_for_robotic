###service.py
import os
import cv2
import base64
import numpy as np
import time
from flask import Flask, request, jsonify, g
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from services.object_detection_ms import MicroserviceObject
from services.object_detection_cloud import CloudModelConnector
from services.object_detection_edge import LocalModelManager
from services.ob_emulator import OBEmulator

from utils.logging_util import setup_logger, get_logger_for_actor
from utils.utility import convert_to_serializable

# Prometheus Metrics
REQUEST_COUNT = Counter('service_requests_total', 'Total number of requests',
						['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('service_request_latency_seconds',
							'Latency of requests in seconds',
							['method', 'endpoint'])
ROBOT_LATENCY = Histogram('robot_request_latency_seconds',
						  'Latency of requests per robot in seconds',
						  ['robot_id'])
EXCEPTIONS_COUNT = Counter('service_exceptions_total',
						   'Total number of exceptions',
						   ['method', 'endpoint', 'exception_type'])

# Directory where debug images might be saved
image_dir = './images'
Microservices_Flag = False
Cloud_Flag = True
Debug = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')


class Service:
	def __init__(self, host='0.0.0.0', port=8081):
		self.app = Flask(__name__)
		self.host = host
		self.port = port

		# Instantiate one MicroserviceObject per model quality

		if Microservices_Flag:
			self.od_micros = {
				"low":    MicroserviceObject(model_quality="low",    confidence_threshold=0.3),
				"medium": MicroserviceObject(model_quality="medium", confidence_threshold=0.5),
				"high": MicroserviceObject(model_quality="high",   confidence_threshold=0.9),
			}
		elif Cloud_Flag:
			self.workload_cloud = CloudModelConnector(num_workers=3)
		else:
			# create a queue-based manager with N worker threads.
			self.model_manager = LocalModelManager(num_workers=3)

		self.ob_emulator = OBEmulator()
		self.emulator_f = os.getenv('EMULATOR', 'False').lower() in ('true', '1', 'yes')

		# Define routes
		self.app.add_url_rule('/services', 'service_detect_objects',
							  self.service_detect_objects, methods=['POST'])
		self.app.add_url_rule('/metrics', 'metrics', self.metrics)

		# Register middleware functions
		self.app.before_request(self.start_timer)
		self.app.after_request(self.record_metrics)

		# Initialize logger
		self.logger = setup_logger(f"{__name__}.Service")

	def start_timer(self):
		"""Start a timer at the beginning of the request."""
		g.start_time = time.time()
		if not hasattr(g, 'sequence_num'):
			g.sequence_num = 0  # Initialize `sequence_num` for the current request

	def record_metrics(self, response):
		"""Record metrics for the request."""
		latency = time.time() - g.start_time
		REQUEST_COUNT.labels(request.method, request.path,
							 str(response.status_code)).inc()
		REQUEST_LATENCY.labels(request.method, request.path).observe(latency)

		# Track latency for each robot
		robot_id = getattr(g, 'actor_id', 'unknown')  # Default if not set
		ROBOT_LATENCY.labels(robot_id).observe(latency)

		self.logger.info(f">> Request to {request.path} from robot {robot_id} took {latency:.4f} seconds")

		# Include optional debugging info in headers
		response.headers["OD-Request-Latency"] = f"{latency:.4f} seconds"
		response.headers["Robot-ID"] = robot_id
		if not hasattr(g, 'sequence_num'):
			g.sequence_num = 0
		response.headers["Sequence-Num"] = g.sequence_num

		return response

	def metrics(self):
		"""Expose Prometheus metrics."""
		return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

	def extract_data_info(self, data):
		"""Extract and validate image and metadata from the request JSON."""
		if not data or 'image_data' not in data:
			return {"error": "No image data provided", "status": 401}

		try:
			image = cv2.imdecode(np.frombuffer(base64.b64decode(data['image_data']), np.uint8),
								 cv2.IMREAD_COLOR)
			if image is None:
				raise ValueError("Image decoding failed")
		except Exception as e:
			return {"error": f"Failed to decode image: {str(e)}", "status": 402}

		# The robot's ID (actor_id), label_id, etc.
		g.actor_id = data.get("actor_id", "robot100")
		try:
			g.label_id = int(data.get('label_id', 0))
		except ValueError:
			return {"error": "Invalid label_id provided", "status": 403}
		g.image_file_name = data.get("image_file_name", f"{g.actor_id}_image")
		g.sequence_num = data.get("sequence_num", 1)

		return {"image": image}

	def service_detect_objects(self):
		"""Handles object detection requests."""
		request_logger = None
		try:
			data = request.get_json()
			extracted_data = self.extract_data_info(data)
			request_logger = get_logger_for_actor(getattr(g, 'actor_id', 'robot100'))

			request_logger.info("Logging specifically for this actor_id")
			request_logger.debug(f"<<< From {g.actor_id} with Seq({g.sequence_num}) "
								 f"and label({g.label_id})")

			if "error" in extracted_data:
				request_logger.debug(f"#10. service_detect_objects ended with an error, "
									 f"{extracted_data['error']}")
				return jsonify({"error": extracted_data["error"]}), extracted_data["status"]

			# Build an image filename (for debug or logging)
			current_time = datetime.now()
			readable_date = current_time.strftime('%d_%H_%M_%S_%f')
			image_name = f"{image_dir}/debug_image_ID({g.label_id})_{readable_date}.jpg"

			image = extracted_data["image"]

			# ------------------------------------------------------------------
			# (A) Decide which model to use based on an LB/Orchestrator header
			# ------------------------------------------------------------------
			# The quality is determined by the requester or a fallback.
			requested_quality = data.get('model_quality', 'low')

			# Hard-coded overrides for demonstration
			if g.actor_id == "robot7":  requested_quality = "low"
			elif g.actor_id == "robot9":  requested_quality = "medium"
			elif g.actor_id == "robot19": requested_quality = "high"
			elif g.actor_id == "robot20": requested_quality = "low"
			elif g.actor_id == "robot21": requested_quality = "low"
			else: requested_quality = "low"

			request_logger.info(f"Using model quality = {requested_quality} for {g.actor_id}")

			# ---------------------------------------
			# Call the model: Microservice OR Local
			# ---------------------------------------
			if Microservices_Flag:
				# Get the MicroserviceObject instance
				microservice_object_instance = self.od_micros.get(requested_quality.lower(), self.od_micros["low"])
				all_detections = microservice_object_instance.detect_objects(image)

			elif Cloud_Flag:
				# It is for one Flask instance per each request by "cloud_detect_objects_async"
				future = self.workload_cloud.cloud_detect_objects_async(image, requested_quality.lower(), g.actor_id, g.sequence_num)
				result = future.result(timeout=10.0)
				actor_id = result['actor_id']
				sequence_num = result['sequence_num']
				all_detections = result['detections']

				if Debug: self.logger.debug(f"<<<<<<service_detect_objects: ID({actor_id})-SN({sequence_num})")

			else:
				# =========================================================
				# Queue-based LocalModelManager asynchronously
				# =========================================================				
				future = self.model_manager.detect_objects_async(image, requested_quality)
				# We block here to retrieve the result before returning
				all_detections = future.result(timeout=10.0)  # optional timeout

			# Distinguish if label_id == 404 => keep all; otherwise filter
			label_id = g.label_id
			if not Cloud_Flag: 
				actor_id = g.actor_id
				sequence_num = g.sequence_num

			detected_objects = []
			for detection in all_detections:
				if detection["label"] == label_id:
					x_box, y_box, w_box, h_box = detection["bbox"]
					center_x = x_box + w_box / 2.0
					center_y = y_box + h_box / 2.0
					detection.update({"x": center_x, "y": center_y, "z": 0.0})
					detected_objects.append(detection)
					request_logger.info(
						f"OD-{actor_id}-label_id({label_id})-image_name({image_name}), "
						f"X:({center_x:.2f}), Y:({center_y:.2f})"
					)

			# Determine the "best" detection by confidence
			best_object = {"error": "No object detected"}
			if detected_objects:
				max_confidence = max(obj.get('confidence', 0) for obj in detected_objects)
				best_objects = [obj for obj in detected_objects if obj.get('confidence', 0) == max_confidence]
				best_object = best_objects[0]  # Ensure only one object is selected
				request_logger.info(
					f"With-OD-{actor_id}-sq(#{sequence_num})-label({best_object.get('label')}), "
					f"X:({best_object['x']:.2f}), Y:({best_object['y']:.2f})"
				)
			elif self.emulator_f:
				# Fallback to emulator if no objects were detected
				emulated_detection = self.ob_emulator.perform_object_detection()
				if emulated_detection:
					best_object = emulated_detection
					request_logger.info(
						f"Without-OD-{actor_id}-sq(#{sequence_num})-label({emulated_detection.get('label')}), "
						f"X:({emulated_detection['x']:.2f}), Y:({emulated_detection['y']:.2f})"
					)

			return jsonify({"objects": convert_to_serializable(best_object)}), 200

		except Exception as e:
			# Record exception metrics
			EXCEPTIONS_COUNT.labels(request.method, request.path, type(e).__name__).inc()
			if request_logger:
				request_logger.error(f"Exception occurred: {str(e)} regarding ID({actor_id})-sq(#{sequence_num})")
			return jsonify({"error": "Internal server error", "details": str(e)}), 500

	def run(self):
		"""Run the Flask app."""
		try:
			self.app.run(host=self.host, port=self.port)
		finally:
			if not Microservices_Flag and self.model_manager:
				self.model_manager.shutdown()


if __name__ == "__main__":
	service = Service()
	service.run()
