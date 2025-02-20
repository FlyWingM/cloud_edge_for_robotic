import pandas as pd
Debug_1 = True

class TaskHandler:
	def __init__(self, params, task_file_path):
		data = pd.read_csv(task_file_path)
		for i in range(len(data)):
			params.task_queue[i] = {"color": data.iloc[i, 0]}

		if Debug_1:
			print(params.task_queue)

		self.task_complete_location = {
			'label': 'box',
			'confidence': 0.0,
			'x': 0.0,
			'y': 0.0,
			'width': 2,
			'height': 2,
			'rotation': 0
		}
