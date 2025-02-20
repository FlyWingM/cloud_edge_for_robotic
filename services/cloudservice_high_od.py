###cloudservice_high_od.py
import os
import base64
import logging
import numpy as np
import cv2
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
    Loads a single PyTorch model (Faster R-CNN by default)
    and offers detect_cloud_objects() for inference on an image (BGR numpy array).
    """
    def __init__(self, model_path, confidence_threshold=0.3):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.torch_model = None

        if Debug:
            logger.info(f"Initializing ObjectDetectorCloud with model_path={model_path}")

        self._load_pytorch_model()

    def _load_pytorch_model(self):
        """
        Loads a PyTorch checkpoint into a Faster R-CNN ResNet-50 FPN model
        or a generic model if needed.
        """
        try:
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))

            # Check if it’s a Faster R-CNN
            if "faster_rcnn" in self.model_path.lower():
                from torchvision.models.detection import fasterrcnn_resnet50_fpn
                import torchvision.transforms as T

                if Debug:
                    logger.info("Detected Faster R-CNN model. Initializing architecture...")
                self.torch_model = fasterrcnn_resnet50_fpn(pretrained=False)

                # Load state dict from 'model' key or directly
                if "model" in checkpoint:
                    self.torch_model.load_state_dict(checkpoint["model"])
                else:
                    self.torch_model.load_state_dict(checkpoint)

                self.torch_model.eval()
                logger.info("Faster R-CNN model loaded successfully.")

                # Define a transform for detection
                self.pytorch_transform = T.Compose([
                    T.ToTensor(),  # Convert image to PyTorch tensor
                ])
            else:
                # If not a recognized Faster R-CNN checkpoint, just load as a generic model
                self.torch_model = checkpoint
                self.torch_model.eval()
                logger.info("Generic PyTorch model loaded successfully.")

        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load PyTorch model: {e}")

    def detect_cloud_objects(self, image_bgr):
        """
        Performs detection on a BGR numpy array using the loaded PyTorch model.
        Returns a list of detection dicts with { "label", "confidence", "bbox" } in normalized coords.
        """
        # If we’re using a known transform for e.g. Faster R-CNN, else define it
        if not hasattr(self, 'pytorch_transform'):
            from torchvision import transforms as T
            self.pytorch_transform = T.Compose([T.ToTensor()])

        # Convert BGR -> RGB
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self.pytorch_transform(rgb_image)
        image_tensor = image_tensor.unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            results = self.torch_model(image_tensor)

        detections = []
        h, w = image_bgr.shape[:2]
        for i in range(len(results[0]["boxes"])):
            score = results[0]["scores"][i].item()
            if score >= self.confidence_threshold:
                x_min, y_min, x_max, y_max = results[0]["boxes"][i].tolist()

                # Convert to normalized coords
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


# ------------------------------------------------
# CloudModelManager (single-model version)
# ------------------------------------------------
import queue
import time
from concurrent.futures import Future
import threading

class CloudModelManager:
    """
    Manages a single ObjectDetectorCloud instance (e.g., for a high-quality model).
    Provides asynchronous detection using a worker queue.
    """
    MODEL_PATH = "./od_models/faster_rcnn_R_50_FPN_3x.pth"

    def __init__(self, num_workers=3):
        self.detector = None
        try:
            self.detector = ObjectDetectorCloud(self.MODEL_PATH)
            logger.info("Loaded 'high' model successfully.")
        except ModelNotLoadedError as e:
            logger.error(f"Failed to load 'high' model: {e}")
            raise

        # A thread-safe queue for incoming tasks
        self._task_queue = queue.Queue()

        # Start worker threads
        self._stop_event = threading.Event()
        self._workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)

    def _worker_loop(self):
        """
        Continuously pulls tasks from the queue and processes them.
        Each task is: (image, future)
        """
        while not self._stop_event.is_set():
            try:
                # Blocks until a task is available or times out
                image, actor_id, sequence_num, future = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue  # Check again if we should stop

            try:
                if not self.detector:
                    raise ModelNotLoadedError("Detector not loaded.")

                future.set_running_or_notify_cancel()
                detections = self.detector.detect_cloud_objects(image)

                if Debug:
                    logger.info(f"_worker_loop processed #{sequence_num} image of ID({actor_id}) detections: {detections}")

                future.set_result({
                    "actor_id": actor_id,
                    "sequence_num": sequence_num,
                    "detections": detections
                })

            except Exception as e:
                logger.error(f"Error processing detection: {e}")
                future.set_exception(e)
            finally:
                self._task_queue.task_done()

    def detect_objects_async(self, image, actor_id, sequence_num):
        """
        Enqueue a request for object detection on a single image.
        Returns a Future that will contain the detection results.
        """
        future = Future()
        if image is not None:
            self._task_queue.put((image, actor_id, sequence_num, future))
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

app = Flask(__name__)

FLASK_PORT = int(os.getenv("FLASK_PORT", "8081"))

# Instantiate the single-model Manager globally
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
         "image_data": <base64-encoded image>
      }
    """
    if detector_cmm is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "image_data" not in data:
        return jsonify({"error": "Missing image_data"}), 400

    try:
        img_data = base64.b64decode(data["image_data"])
        np_img = np.frombuffer(img_data, np.uint8)
        image_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        actor_id = data.get('actor_id', 'robot0')
        sequence_num = data.get('sequence_num', 0)

        if image_bgr is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as e:
        return jsonify({"error": f"Failed to decode image_data: {str(e)}"}), 400

    if Debug: logger.info(f"<<<<<<detect_cloud from ({actor_id}) with ({sequence_num})")

    # Detect objects (async, then wait result)
    future = detector_cmm.detect_objects_async(image_bgr, actor_id, sequence_num)
    result = future.result(timeout=60.0)  # optional timeout
    detections = result['detections']
    actor_id = result['actor_id']
    sequence_num = result['sequence_num']

    if Debug: logger.info(f">>>>>>detect_cloud from ({actor_id}) with ({sequence_num}), detections:{detections}")
    return jsonify({"detections": detections})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=FLASK_PORT)
