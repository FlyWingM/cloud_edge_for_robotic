###cloudservice_od.py
import os
import base64
import logging
import numpy as np
import cv2
import tensorflow as tf
import torch
from flask import Flask, request, jsonify

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LocalDebug")
Debug = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')

# Exceptions
class ModelNotLoadedError(Exception):
	"""Exception raised if the ML model is not loaded properly."""
	pass

class ObjectDetectorCloud:
	"""
	Loads either a TFLite or PyTorch model (depending on file extension) and
	offers detect_cloud_objects() to run inference on an image.
	"""
	def __init__(self, model_path, confidence_threshold=0.3):
		self.model_path = model_path
		self.confidence_threshold = confidence_threshold

		self.framework = None     # 'tflite' or 'pytorch'
		self.interpreter = None   # For TFLite
		self.torch_model = None   # For PyTorch

		if Debug: logger.info(f"Initializing ObjectDetectorCloud with model_path={model_path}")

		self._load_model()

	def _load_model(self):
		ext = os.path.splitext(self.model_path)[1].lower()
		if ext == '.tflite':
			if Debug: logger.info(f"Loading TensorFlow Lite model from: {self.model_path}")
			self._load_tflite_model()
			self.framework = 'tflite'
		elif ext in ('.pt', '.pth'):
			if Debug: logger.info(f"Loading PyTorch model from: {self.model_path}")
			self._load_pytorch_model()
			self.framework = 'pytorch'
		else:
			raise ModelNotLoadedError(
				f"Unsupported model extension '{ext}'. Must be .tflite, .pt, or .pth."
			)

	def _load_tflite_model(self):
		try:
			interpreter = tf.lite.Interpreter(model_path=self.model_path)
			interpreter.allocate_tensors()
			self.interpreter = interpreter

			input_details = interpreter.get_input_details()
			output_details = interpreter.get_output_details()

			if Debug:
				logger.info("TFLite Model Input Details:")
				for detail in input_details:
					logger.info(detail)

				logger.info("\nTFLite Model Output Details:")
				for detail in output_details:
					logger.info(detail)

				logger.info("TFLite model loaded successfully.")
		except Exception as e:
			raise ModelNotLoadedError(f"Failed to load TFLite model: {e}")

	def _load_pytorch_model(self):
		try:
			checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
			# Attempt to see if there's a known structure
			if "faster_rcnn" in self.model_path.lower():
				from torchvision.models.detection import fasterrcnn_resnet50_fpn
				import torchvision.transforms as T

				if Debug:
					logger.info("Detected Faster R-CNN model. Initializing architecture...")
				self.torch_model = fasterrcnn_resnet50_fpn(pretrained=False)
				
				# If the checkpoint has "model" key, load from that, else assume direct state dict
				if "model" in checkpoint:
					self.torch_model.load_state_dict(checkpoint["model"])
				else:
					self.torch_model.load_state_dict(checkpoint)

				self.torch_model.eval()
				logger.info("Faster R-CNN model loaded successfully.")

				# Define an internal transform for detection
				self.pytorch_transform = T.Compose([
					T.ToTensor(),  # Convert image to PyTorch tensor
				])
			else:
				# Load as a generic PyTorch model
				self.torch_model = checkpoint
				self.torch_model.eval()
				logger.info("Generic PyTorch model loaded successfully.")

		except Exception as e:
			raise ModelNotLoadedError(f"Failed to load PyTorch model: {e}")

	def detect_cloud_objects(self, image_input):
		"""
		image_input can be either a TF tensor (for TFLite) or 
		a raw BGR NumPy array. We will handle both scenarios:
		 - If the model is TFLite, we expect a TF tensor (H,W,3) or (1,H,W,3).
		 - If the model is PyTorch, we can convert a TF tensor to NumPy or 
		   directly accept a BGR NumPy image.
		"""
		if self.framework == 'tflite':
			# If we received a plain NumPy BGR image, convert to TF Tensor
			if isinstance(image_input, np.ndarray):
				image_input = self._bgr_to_tf_tensor(image_input)
			return self._detect_objects_tflite(image_input)
		elif self.framework == 'pytorch':
			# If we received a TF tensor, convert to NumPy BGR
			if isinstance(image_input, tf.Tensor):
				image_input = image_input.numpy().astype(np.uint8)
				image_input = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
			return self._detect_objects_pytorch(image_input)
		else:
			raise ModelNotLoadedError("Model not loaded or unknown framework.")

	# ---------------
	# TFLite Methods
	# ---------------
	def _detect_objects_tflite(self, image_tf):
		# Dispatch to the correct TFLite postprocessing
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
		"""
		TFLite SSD MobileNet detection logic
		"""
		interpreter = self.interpreter
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		
		_, input_height, input_width, _ = input_details[0]['shape']

		# Preprocess (resize, batch dimension, etc.)
		resized = tf.image.resize(image_tf, (input_width, input_height))
		resized = resized[tf.newaxis, :]  # shape (1, H, W, 3)

		if input_details[0]['dtype'] == np.uint8:
			resized = ((resized / 255.0) * 127.5 + 127.5).numpy().astype(np.uint8)
		else:
			resized = resized.numpy().astype(np.float32)

		self._set_input_tensor(interpreter, resized)
		interpreter.invoke()

		boxes_tensor = self._get_output_tensor(interpreter, 0)[0]  # (num_boxes, 4)
		class_ids_tensor = self._get_output_tensor(interpreter, 1)[0]
		scores_tensor = self._get_output_tensor(interpreter, 2)[0]
		num_detections_tensor = self._get_output_tensor(interpreter, 3)
		count = int(num_detections_tensor.flatten()[0])

		detections = []
		for i in range(count):
			score = float(scores_tensor[i])
			if score >= self.confidence_threshold:
				class_id = int(class_ids_tensor[i])
				ymin, xmin, ymax, xmax = boxes_tensor[i]
				# Clip and convert to width/height
				y_min = max(0.0, ymin)
				x_min = max(0.0, xmin)
				y_max = min(1.0, ymax)
				x_max = min(1.0, xmax)
				w_norm = x_max - x_min
				h_norm = y_max - y_min

				detections.append({
					"label": class_id,
					"confidence": score,
					"bbox": (float(x_min), float(y_min), float(w_norm), float(h_norm))
				})
		return detections

	def _detect_objects_tflite_yolov5(self, image_tf):
		"""
		Example YOLOv5 TFLite logic
		"""
		interpreter = self.interpreter
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		_, input_height, input_width, _ = input_details[0]['shape']
		resized = tf.image.resize(image_tf, (input_height, input_width))
		resized = resized[tf.newaxis, :]

		if input_details[0]['dtype'] == np.uint8:
			resized = ((resized / 255.0) * 127.5 + 127.5).numpy().astype(np.uint8)
		else:
			resized = resized.numpy().astype(np.float32)

		self._set_input_tensor(interpreter, resized)
		interpreter.invoke()

		outputs = self._get_output_tensor(interpreter, 0)  # shape [1, N, 6]
		outputs = np.squeeze(outputs, axis=0)             # shape [N, 6]

		detections = []
		for det in outputs:
			x_min, y_min, x_max, y_max, score, class_id = det[:6]
			if score >= self.confidence_threshold:
				# Ensure coords are within [0, 1] if the model is scaled that way
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

		# Optionally apply non_max_suppression if needed
		# detections = self.non_max_suppression(detections, iou_threshold=0.02)

		return detections

	def _detect_objects_tflite_cube(self, image_tf):
		"""
		Example TFLite inference for a custom model: 'model_cube.tflite'
		"""
		# If image_tf is a TF tensor, convert to NumPy
		if isinstance(image_tf, tf.Tensor):
			np_img = image_tf.numpy().astype(np.uint8)
		else:
			np_img = image_tf

		# Convert np_img to the correct size/dtype, etc.
		interpreter = self.interpreter
		input_shape = interpreter.get_input_details()[0]['shape']
		_, model_height, model_width, _ = input_shape
		
		# Convert if needed
		if np_img.shape[:2] != (model_height, model_width):
			np_img = cv2.resize(np_img, (model_width, model_height))

		# If model expects float32, must normalize
		dtype_expected = interpreter.get_input_details()[0]['dtype']
		np_img = self._convert_dtype_for_tflite(np_img, dtype_expected)

		# Set and invoke
		self._set_input_tensor(interpreter, np_img)
		interpreter.invoke()

		# Retrieve outputs (assuming "class_ids", "scores", "bboxes")
		inference_info = self._get_inference_info()
		if not inference_info:
			raise ValueError("Failed to retrieve inference results (cube model).")

		detections = []
		for i, score in enumerate(inference_info["scores"]):
			if score >= self.confidence_threshold:
				x_min, y_min, x_max, y_max = inference_info["bboxes"][i]
				w_norm = x_max - x_min
				h_norm = y_max - y_min
				detections.append({
					"label": int(inference_info["class_ids"][i]),
					"confidence": float(score),
					"bbox": (float(x_min), float(y_min), float(w_norm), float(h_norm))
				})
		return detections

	def _convert_dtype_for_tflite(self, image, expected_dtype):
		"""
		Helper to convert image to the right dtype (UINT8 or FLOAT32).
		Expands dims for batch (1, H, W, C).
		"""
		if expected_dtype == np.uint8:
			image = image.astype(np.uint8)
		elif expected_dtype == np.float32:
			image = image.astype(np.float32)
			image /= 255.0
		else:
			raise ValueError(f"Unexpected TFLite input dtype: {expected_dtype}")
		return np.expand_dims(image, axis=0)  # (1, H, W, C)

	def _get_inference_info(self):
		"""
		Example helper that attempts to retrieve [class_ids, scores, bboxes].
		Adjust to match your model's actual output indexes.
		"""
		try:
			class_id_tensor = self._get_output_tensor(self.interpreter, 0)
			bbox_tensor     = self._get_output_tensor(self.interpreter, 1)
			score_tensor    = self._get_output_tensor(self.interpreter, 3)

			class_ids = class_id_tensor.flatten().tolist()
			scores    = score_tensor.flatten().tolist()
			
			# If bounding boxes are shape (1, num_boxes, 4), flatten them
			if len(bbox_tensor.shape) == 3 and bbox_tensor.shape[-1] == 4:
				bboxes = bbox_tensor.reshape(-1, 4).tolist()
			else:
				logger.warning(f"Unexpected bbox_tensor shape: {bbox_tensor.shape}")
				bboxes = []

			return {
				"class_ids": class_ids,
				"scores": scores,
				"bboxes": bboxes
			}
		except Exception as e:
			logger.error(f"Error retrieving inference output: {e}")
			return None

	# ---------------
	# PyTorch Method
	# ---------------
	def _detect_objects_pytorch(self, image_bgr):
		"""
		If it's a Faster R-CNN model, we do the typical detection flow.
		"""
		if not hasattr(self, 'pytorch_transform'):
			# Fallback if not a Faster R-CNN, or define your transform
			from torchvision import transforms as T
			self.pytorch_transform = T.Compose([T.ToTensor()])

		# Convert BGR -> RGB
		rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		image_tensor = self.pytorch_transform(rgb_image)
		image_tensor = image_tensor.unsqueeze(0)  # batch dimension

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

	# ------------------------
	# TFLite Helper Get/Set
	# ------------------------
	def _set_input_tensor(self, interpreter, image):
		input_details = interpreter.get_input_details()[0]
		interpreter.set_tensor(input_details['index'], image)

	def _get_output_tensor(self, interpreter, index):
		output_details = interpreter.get_output_details()[index]
		return interpreter.get_tensor(output_details['index']).copy()

	def _bgr_to_tf_tensor(self, bgr_image):
		"""
		Helper to convert a BGR (NumPy) image to a TF tensor (RGB).
		"""
		rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
		image_tf = tf.convert_to_tensor(rgb_image, dtype=tf.float32)
		return image_tf


# ------------------------------------------------
# CloudModelManager
# ------------------------------------------------
import queue
import time
from concurrent.futures import Future
import threading

class CloudModelManager:
	"""
	Manages three ObjectDetectorCloud instances for 'low', 'medium', 'high'.
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
				self.detectors[quality] = ObjectDetectorCloud(path)
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
					detector = self.detectors.get("high", None)

				future.set_running_or_notify_cancel()
				detections = detector.detect_cloud_objects(image)

				if Debug: logger.info(f"_worker_loop, detections({detections})")

				future.set_result(detections)
			except Exception as e:
				logger.error(f"Error processing detection: {e}")
				future.set_exception(e)
			finally:
				# Mark this task as done to unblock queue
				self._task_queue.task_done()


	def detect_objects_async(self, image, model_quality="high"):
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
		if Debug: logger.info(f"_task_queue, image-len({len(image)}), model_quality({model_quality.lower()}) with future")
		return future

	def shutdown(self):
		"""
		Signals the worker threads to exit and waits for them to finish.
		"""
		logger.info("Shutting down CloudModelManager...")

		# Signal stop
		self._stop_event.set()

		# Wait for all tasks to complete
		self._task_queue.join()

		# Join worker threads
		for t in self._workers:
			t.join()
		logger.info("CloudModelManager shutdown complete.")

# ------------------------------------------------
# Flask App for HTTP-based detection
# ------------------------------------------------
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Example environment-based configdetect_objects
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))
FLASK_PORT = int(os.getenv("FLASK_PORT", "8081"))

# Instantiate the model/Detector once globally
try:
	detector_cmm = CloudModelManager()
except ModelNotLoadedError as ex:
	logger.error(f"Failed to load model: {ex}")
	detector_cmm = None


@app.route("/detect_cloud", methods=["POST"])
def detect_cloud():
	"""
	Example endpoint:
	Expects JSON with:
	  {
		 "image_data": <base64-encoded image>,
		 "confidence_threshold": 0.5 (optional)
	  }
	"""
	if detector_cmm is None:
		return jsonify({"error": "Model not loaded"}), 500

	data = request.get_json()
	if not data or "image_data" not in data:
		return jsonify({"error": "Missing image_data"}), 400

	if "model_quality" in data:
		requested_quality = data["model_quality"]
	else:
		requested_quality = 'high'

	try:
		img_data = base64.b64decode(data["image_data"])
		np_img = np.frombuffer(img_data, np.uint8)
		image_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
		if image_bgr is None:
			raise ValueError("cv2.imdecode returned None")
	except Exception as e:
		return jsonify({"error": f"Failed to decode image_data: {str(e)}"}), 400

	future = detector_cmm.detect_objects_async(image_bgr, requested_quality)
	detections = future.result(timeout=60.0)  # optional timeout

	return jsonify({"detections": detections})

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=FLASK_PORT)