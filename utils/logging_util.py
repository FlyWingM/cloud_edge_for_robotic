# logging_util.py

import logging
import os
from datetime import datetime

class CustomFormatter(logging.Formatter):
	"""Custom formatter for adding milliseconds to logs."""
	def formatTime(self, record, datefmt=None):
		"""
		Override formatTime to allow '%f' in the date format
		for millisecond display, e.g., '%d_%H-%M-%S-%f'.
		"""
		try:
			ct = datetime.fromtimestamp(record.created)
			if datefmt:
				s = ct.strftime(datefmt)
				# Replace %f in the date format with the actual milliseconds
				if "%f" in datefmt:
					s = s.replace("%f", f"{int(record.msecs):03}")
				return s
			else:
				t = ct.strftime("%Y-%m-%d %H:%M:%S")
				return f"{t}.{int(record.msecs):03}"
		except Exception as e:
			print(f"DEBUG: Error formatting time: {e}")
			raise


def setup_logger(
	name,
	log_level=logging.DEBUG,
	date_format='%d_%H-%M-%S-%f'
):
	"""
	Set up a logger with the specified name, level, and a custom formatter.
	By default, logs go to the console (StreamHandler).
	"""
	logger = logging.getLogger(name)
	logger.setLevel(log_level)

	if not logger.hasHandlers():
		handler = logging.StreamHandler()
		handler.setLevel(log_level)
		log_format = '%(name)s - %(levelname)s - %(message)s at %(asctime)s'
		formatter = CustomFormatter(log_format, datefmt=date_format)
		handler.setFormatter(formatter)
		logger.addHandler(handler)

	return logger


def get_logger_for_actor(
	actor_id,
	log_level=logging.DEBUG,
	date_format='%d_%H-%M-%S-%f'
):
	"""
	Return a logger dedicated to a specific actor_id, writing logs to a file
	named after the actor. This reuses the CustomFormatter for timestamp
	formatting and includes milliseconds in log messages.
	"""
	# Each actor uses a unique logger name to ensure separate handlers.
	logger_name = f"actor_{actor_id}"
	logger = logging.getLogger(logger_name)
	logger.setLevel(log_level)

	# Create the logs directory if it doesn't already exist
	os.makedirs("./logs", exist_ok=True)
	log_file_path = f"./logs/{actor_id}.log"

	# Only add a FileHandler if none exists (avoid duplicate logs)
	if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
		file_handler = logging.FileHandler(log_file_path)
		file_handler.setLevel(log_level)
		log_format = '%(name)s - %(levelname)s - %(message)s at %(asctime)s'
		formatter = CustomFormatter(log_format, datefmt=date_format)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

	return logger
