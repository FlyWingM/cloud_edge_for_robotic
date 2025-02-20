### ob_emulator.py
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import random
import time

object_information_file_path = f'./task/object_information.csv'

class OBEmulator:

	def __init__(self):
		self.object_info = {}
		self.sample_size = 0

		data = pd.read_csv(object_information_file_path)
		self.sample_size = len(data)
		# Initialize an empty dictionary to store object information
		self.object_info = {}

		# Iterate over the rows in the data
		for i in range(len(data)):
			self.object_info[i] = {
				"label": data.iloc[i, 0],
				"confidence": data.iloc[i, 1],
				"x": data.iloc[i, 2],
				"y": data.iloc[i, 3],
				"z": data.iloc[i, 4],
				"width": data.iloc[i, 5],
				"height": data.iloc[i, 6],
				"rotation": data.iloc[i, 7]
			}

	def perform_object_detection(self):
		"""
		Placeholder function for object detection logic.
		Replace this function with your actual object detection code.
		"""
		time.sleep(random.uniform(0.5, 1))

		sample_num = random.randint(0, self.sample_size-1)
		object_data = self.object_info[sample_num]

		# Convert any numpy types to native Python types
		for key, value in object_data.items():
			if isinstance(value, (np.int64, np.float64)):  # Check for numpy types
				object_data[key] = value.item()  # Convert to native Python type

		return object_data

