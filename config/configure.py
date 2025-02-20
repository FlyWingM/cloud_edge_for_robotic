# coding: utf-8?
# ===================================================================

import os

class Parameters:
	def __init__(self):
		os.environ['CLOUDGRIPPER_TOKEN'] = 'InpNY3tiiphZrGqA9olNTHHvXBhCqfjz'
		self.token = 'InpNY3tiiphZrGqA9olNTHHvXBhCqfjz'
		#self.robotName = ['robot7', 'robot9', 'robot19', 'robot20', 'robot21']
		self.robotName = ['robot9']

		self.command = {
			"x": 0,
			"y": 0,
			"rotation": 0,
		}	

		self.task_queue = {}


