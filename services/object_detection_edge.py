### object_detection_edge.py
import os
import logging
import threading
import numpy as np
import cv2
import tensorflow as tf
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Exceptions
class ModelNotLoadedError(Exception):
	"""Exception raised if the ML model is not loaded properly."""
	pass

# ObjectDetector Class
class ObjectDetector:
	"""
	Loads either a TFLite or PyTorch model (depending on file extension) and
	offers detect_objects() to run inference on an image.
	"""
	def __init__(self, model_path, confidence_threshold=0.3):
		self.model_path = model_path
		self.confidence_threshold = confidence_threshold
		self.framework = None     # 'tflite' or 'pytorch'
		self.interpreter = None   # For TFLite
		self.torch_model = None   # For PyTorch
		self.pytorch_transform = None

		# Add a lock to ensure thread-safe inference
		self._lock = threading.Lock()

		logger.info(f"Initializing ObjectDetector with model_path={model_path}")
		self._load_model()

	def _load_model(self):
		ext = os.path.splitext(self.model_path)[1].lower()
		if ext == '.tflite':
			logger.info(f"Loading TFLite model from: {self.model_path}")
			self._load_tflite_model()
			self.framework = 'tflite'
		elif ext in ('.pt', '.pth'):
			logger.info(f"Loading PyTorch model from: {self.model_path}")
			self._load_pytorch_model()
			self.framework = 'pytorch'
		else:
			raise ModelNotLoadedError(
				f"Unsupported model extension '{ext}'. Must be .tflite, .pt, or .pth."
			)

	# TFLite Loading ---------------
	def _load_tflite_model(self):
		try:
			interpreter = tf.lite.Interpreter(model_path=self.model_path)
			interpreter.allocate_tensors()
			self.interpreter = interpreter
			logger.info("TFLite model loaded successfully.")
		except Exception as e:
			raise ModelNotLoadedError(f"Failed to load TFLite model: {e}")

	# PyTorch Loading ---------------
	def _load_pytorch_model(self):
		try:
			checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))

			# Heuristic: if this is a Faster R-CNN model
			if "faster_rcnn" in self.model_path.lower():
				logger.info("Detected Faster R-CNN model. Initializing architecture...")
				self.torch_model = fasterrcnn_resnet50_fpn(pretrained=False)

				# Some checkpoints might contain "model" key
				if "model" in checkpoint:
					self.torch_model.load_state_dict(checkpoint["model"])
				else:
					self.torch_model.load_state_dict(checkpoint)

				self.torch_model.eval()
				logger.info("Faster R-CNN model loaded successfully.")

				# Define transform for detection
				self.pytorch_transform = T.Compose([
					T.ToTensor(),  # Convert image to PyTorch tensor
				])
			else:
				# Load as generic PyTorch model
				self.torch_model = checkpoint
				self.torch_model.eval()
				logger.info("Generic PyTorch model loaded successfully.")

		except Exception as e:
			raise ModelNotLoadedError(f"Failed to load PyTorch model: {e}")

	# --------------- Detect Objects (entry point) ---------------
	def detect_objects(self, image_bgr):
		"""
		:param image_bgr: An OpenCV BGR image (NumPy array).
		:return: list of detections, each detection is a dict with:
				{
					"label": <int>,
					"confidence": <float>,
					"bbox": (x_norm, y_norm, w_norm, h_norm)
				}
		"""
		with self._lock:
			if self.framework == 'tflite':
				return self._detect_objects_tflite(image_bgr)
			elif self.framework == 'pytorch':
				return self._detect_objects_pytorch(image_bgr)
			else:
				raise ModelNotLoadedError("Model not loaded or unknown framework.")

	# TFLite Inference ---------------
	def _detect_objects_tflite(self, image_bgr):
		# Convert BGR -> RGB -> TF Tensor
		image_tf = self._bgr_to_tf_tensor(image_bgr)

		# Dispatch to specific logic based on known TFLite model types
		model_lower = self.model_path.lower()
		if "ssd_mobilenet_v2_coco_select_ops" in model_lower:
			return self._detect_objects_tflite_ssd_mobilenet(image_tf)
		elif "yolov5" in model_lower:
			return self._detect_objects_tflite_yolov5(image_tf)
		elif "model_cube" in model_lower:
			return self._detect_objects_tflite_cube(image_tf)
		else:
			raise ModelNotLoadedError(
				"Unknown TFLite model structure. Please add a specific postprocessing method."
			)

	def _detect_objects_tflite_ssd_mobilenet(self, image_tf):
		interpreter = self.interpreter
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		# Preprocess: resize to model input
		_, input_height, input_width, _ = input_details[0]['shape']
		resized = tf.image.resize(image_tf, (input_width, input_height))
		resized = resized[tf.newaxis, :]  # (1, H, W, 3)

		# Convert data type if model expects uint8
		if input_details[0]['dtype'] == np.uint8:
			resized = ((resized / 255.0) * 127.5 + 127.5).numpy().astype(np.uint8)
		else:
			resized = resized.numpy().astype(np.float32)

		# Run inference
		interpreter.set_tensor(input_details[0]['index'], resized)
		interpreter.invoke()

		boxes_tensor    = interpreter.get_tensor(output_details[0]['index'])[0]  # (N,4)
		class_ids_tensor= interpreter.get_tensor(output_details[1]['index'])[0]
		scores_tensor   = interpreter.get_tensor(output_details[2]['index'])[0]
		num_detections  = interpreter.get_tensor(output_details[3]['index'])[0]

		detections = []
		for i in range(int(num_detections)):
			score = float(scores_tensor[i])
			if score >= self.confidence_threshold:
				class_id = int(class_ids_tensor[i])
				ymin, xmin, ymax, xmax = boxes_tensor[i]
				x_min = max(0.0, xmin)
				y_min = max(0.0, ymin)
				x_max = min(1.0, xmax)
				y_max = min(1.0, ymax)
				w_norm = x_max - x_min
				h_norm = y_max - y_min

				detections.append({
					"label": class_id,
					"confidence": score,
					"bbox": (float(x_min), float(y_min), float(w_norm), float(h_norm))
				})

		return detections

	def _detect_objects_tflite_yolov5(self, image_tf):
		interpreter = self.interpreter
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		_, input_height, input_width, _ = input_details[0]['shape']
		resized = tf.image.resize(image_tf, (input_height, input_width))
		resized = resized[tf.newaxis, :]

		# Convert data type if model expects uint8
		if input_details[0]['dtype'] == np.uint8:
			resized = ((resized / 255.0) * 127.5 + 127.5).numpy().astype(np.uint8)
		else:
			resized = resized.numpy().astype(np.float32)

		interpreter.set_tensor(input_details[0]['index'], resized)
		interpreter.invoke()

		outputs = interpreter.get_tensor(output_details[0]['index'])
		outputs = np.squeeze(outputs, axis=0)  # shape [N, 6]

		detections = []
		for det in outputs:
			x_min, y_min, x_max, y_max, score, class_id = det[:6]
			if score >= self.confidence_threshold:
				x_min = max(0.0, min(1.0, x_min))
				y_min = max(0.0, min(1.0, y_min))
				x_max = max(0.0, min(1.0, x_max))
				y_max = max(0.0, min(1.0, y_max))
				w_norm = max(0.0, x_max - x_min)
				h_norm = max(0.0, y_max - y_min)

				detections.append({
					"label": int(class_id),
					"confidence": float(score),
					"bbox": (float(x_min), float(y_min), float(w_norm), float(h_norm))
				})

		return detections

	def _detect_objects_tflite_cube(self, image_tf):
		"""
		Example TFLite inference for a custom 'model_cube.tflite'.
		Adjust as appropriate for your actual model outputs.
		"""
		interpreter = self.interpreter
		input_details = interpreter.get_input_details()

		_, model_height, model_width, _ = input_details[0]['shape']
		# image_tf is (H, W, 3). Resize
		resized = tf.image.resize(image_tf, (model_height, model_width))

		# Expand dims -> (1, H, W, 3)
		resized = resized[tf.newaxis, :]

		# Convert data type if needed
		if input_details[0]['dtype'] == np.uint8:
			resized = ((resized / 255.0) * 127.5 + 127.5).numpy().astype(np.uint8)
		else:
			resized = resized.numpy().astype(np.float32)

		# Set and invoke
		interpreter.set_tensor(input_details[0]['index'], resized)
		interpreter.invoke()

		# Example: read from known output indexes
		# (You must adjust indexes to your model's real outputs.)
		class_ids_tensor = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
		bboxes_tensor    = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])
		scores_tensor    = interpreter.get_tensor(interpreter.get_output_details()[3]['index'])

		class_ids = class_ids_tensor.flatten().tolist()
		scores    = scores_tensor.flatten().tolist()

		# Reshape bounding boxes if needed
		if len(bboxes_tensor.shape) == 3 and bboxes_tensor.shape[-1] == 4:
			bboxes = bboxes_tensor.reshape(-1, 4).tolist()
		else:
			logger.warning(f"Unexpected bbox_tensor shape: {bboxes_tensor.shape}")
			bboxes = []

		detections = []
		for i, score in enumerate(scores):
			if score >= self.confidence_threshold and i < len(bboxes):
				x_min, y_min, x_max, y_max = bboxes[i]
				w_norm = x_max - x_min
				h_norm = y_max - y_min
				detections.append({
					"label": int(class_ids[i]),
					"confidence": float(score),
					"bbox": (float(x_min), float(y_min), float(w_norm), float(h_norm))
				})
		return detections

	# PyTorch Inference ---------------
	def _detect_objects_pytorch(self, image_bgr):
		if not self.pytorch_transform:
			# If not a known detection model, create default transform
			self.pytorch_transform = T.Compose([T.ToTensor()])

		# Convert BGR -> RGB
		rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		image_tensor = self.pytorch_transform(rgb_image)
		image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)

		with torch.no_grad():
			results = self.torch_model(image_tensor)

		detections = []
		h, w = image_bgr.shape[:2]
		for i in range(len(results[0]["boxes"])):
			score = results[0]["scores"][i].item()
			if score >= self.confidence_threshold:
				x_min, y_min, x_max, y_max = results[0]["boxes"][i].tolist()
				w_norm = (x_max - x_min) / w
				h_norm = (y_max - y_min) / h
				x_min_norm = x_min / w
				y_min_norm = y_min / h

				detections.append({
					"label": int(results[0]["labels"][i].item()),
					"confidence": score,
					"bbox": (x_min_norm, y_min_norm, w_norm, h_norm)
				})
		return detections

	# BGR -> TF Tensor ---------------
	def _bgr_to_tf_tensor(self, bgr_image):
		rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
		image_tf  = tf.convert_to_tensor(rgb_image, dtype=tf.float32)
		return image_tf


import queue
import time
from concurrent.futures import Future

# Local Model Manager
class LocalModelManager:
	"""
	Manages three ObjectDetector instances for 'low', 'medium', 'high'.
	Adjust model paths to your setup.
	"""
	MODEL_PATHS = {
		"low":    "./od_models/model_cube.tflite",
		"medium": "./od_models/yolov5.tflite",
		"high":   "./od_models/faster_rcnn_R_50_FPN_3x.pth"
	}

	def __init__(self, num_workers=3):
		#self.num_workers = 3
		self.detectors = {}
		for quality, path in self.MODEL_PATHS.items():
			try:
				self.detectors[quality] = ObjectDetector(path)
				logger.info(f"Loaded '{quality}' model successfully.")
			except Exception as e:
				logger.error(f"Failed to load '{quality}' model: {e}")
				self.detectors[quality] = None

		# A thread-safe queue for incoming tasks
		self._task_queue = queue.Queue()

		# Start worker threads
		self._stop_event = threading.Event()

		# Start worker threads
		self._workers = []
		for _ in range(num_workers):
			t = threading.Thread(target=self._worker_loop, daemon=True)
			t.start()
			self._workers.append(t)

	def _worker_loop(self):
		"""
		Continuously pulls tasks from the queue and processes them.
		Each task is: (image, model_quality, future)
		"""
		while not self._stop_event.is_set():
			try:
				# Blocks until a task is available or timeout
				image, model_quality, future = self._task_queue.get(timeout=0.5)
			except queue.Empty:
				continue  # check stop_event again

			try:
				detector = self.detectors.get(model_quality, None)
				if detector is None:
					# fallback if not loaded
					detector = self.detectors.get("low", None)
				if detector is None:
					raise ValueError("No valid model detector available.")

				future.set_running_or_notify_cancel()
				detections = detector.detect_objects(image)
				future.set_result(detections)
			except Exception as e:
				logger.error(f"Error processing detection: {e}")
				future.set_exception(e)
			finally:
				# Mark this task as done to unblock queue
				self._task_queue.task_done()


	def detect_objects_async(self, image, model_quality="low"):
		"""
		Enqueue a request for object detection on a single image.
		Returns a Future that will contain the detections once processed.

		:param image: OpenCV BGR image (numpy array)
		:param model_quality: "low" | "medium" | "high"
		:return: concurrent.futures.Future
		"""
		future = Future()
		# Put a tuple in the queue for our workers to pick up
		self._task_queue.put((image, model_quality.lower(), future))
		return future

	def shutdown(self):
		"""
		Signals the worker threads to exit and waits for them to finish.
		"""
		logger.info("Shutting down LocalModelManager...")

		# Signal stop
		self._stop_event.set()

		# Wait for all tasks to complete
		self._task_queue.join()

		# Join worker threads
		for t in self._workers:
			t.join()
		logger.info("LocalModelManager shutdown complete.")

	def detect_objects(self, image, model_quality="low"):
		"""
		Select the proper local ObjectDetector instance and run inference.
		:param image: OpenCV BGR image
		:param model_quality: "low" | "medium" | "high"
		"""
		detector = self.detectors.get(model_quality.lower())
		if detector is None:
			logger.warning(f"No loaded detector for quality '{model_quality}', "
						   "falling back to 'low'.")
			detector = self.detectors["low"]

		return detector.detect_objects(image)

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
