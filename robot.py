### robot.py

import os
import time
import requests
from config.configure import Parameters
from grasping.robot_gripper import PhysicalRobotGripper
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from utils.logging_util import setup_logger
import threading
import signal
from queue import Queue, Empty

Debug_1 = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')
Debug_2 = False

class RobotGripperManager:
    def __init__(self, selected_robot_names=None):
        self.params = Parameters()
        self.token = os.getenv('CLOUDGRIPPER_TOKEN')
        self.object_api_url = os.getenv('DETECTION_API_URL')
        self.robot_names = selected_robot_names  # We'll use only the first name
        self.num_of_requests = 1  # This can be adjusted if needed

        # Exit signal event
        self.exit_signal = threading.Event()

        self.logger = setup_logger(f"{__name__}.RobotGripperManager")
        self.lock = threading.Lock()

        if not self.token:
            raise ValueError("CLOUDGRIPPER_TOKEN is not set in the environment variables.")
        if not self.object_api_url:
            raise ValueError("DETECTION_API_URL is not set in the environment variables.")

        # Only use the first robot name (if any)
        if self.robot_names:
            self.robot_name = self.robot_names[0]
        else:
            self.robot_name = None

        # Initialization of the single robot
        self.robot = self.initialize_robot(self.robot_name)

        # We'll use a ThreadPoolExecutor to actually run the send_image_async calls
        # and keep the concurrency separate from the worker-thread logic.
        self.executor = ThreadPoolExecutor(max_workers=5)

    def set_exit_signal(self, signum, frame):
        """
        Signal handler method that sets an event to stop all tasks gracefully.
        """
        self.logger.info(f"Received signal {signum}. Exiting gracefully.")
        self.exit_signal.set()

    def initialize_robot(self, robot_name):
        """Initializes a PhysicalRobotGripper object for the given robot name."""
        if not robot_name:
            self.logger.warning("No robot name provided for initialization.")
            return None

        try:
            return PhysicalRobotGripper(robot_name, self.token, self.object_api_url)
        except ValueError as e:
            self.logger.error(f"Failed to initialize robot {robot_name}: {e}")
            return None

    def send_image_async(self, robot_name, image, image_file_name):
        try:
            object_location = self.robot.sending_image_and_getting_info_via_comm(
                image,
                image_file_name=image_file_name,
                mode="camera"
            )
            return robot_name, object_location
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout while sending image for robot {robot_name}")
            return robot_name, None
        except Exception as e:
            self.logger.error(f"Unexpected error for robot {robot_name}: {e}")
            return robot_name, None

    def process_robot_tasks(self, robot_name, object_location):
        """
        Processes tasks for the robot with the given object information.
        Here, you can implement post-processing or movement logic.
        """
        try:
            if object_location:
                if Debug_2: 
                    self.logger.debug(f"Object info for {robot_name}: {object_location}")
            else:
                self.logger.warning(f"No objects detected for {robot_name}. Skipping tasks.")
            # Additional tasks or post-processing can go here
            # e.g., self.robot.back_station_robot()
        except Exception as e:
            self.logger.error(f"Error processing tasks for robot {robot_name}: {e}")

    def _worker_loop(self, task_queue, results, results_lock):
        """
        Worker thread function:
         - Continuously fetch completed futures from the queue
         - Wait on future.result() to get the outcome of send_image_async
         - Store or process the results
        """
        while not self.exit_signal.is_set():
            try:
                future = task_queue.get(timeout=1)
            except Empty:
                # No task was available for 1 second; check exit_signal again
                continue

            # If we use a sentinel None to stop workers:
            if future is None:
                # Put it back in case other workers need to see it too
                task_queue.put(None)
                break

            try:
                # Wait for the future to complete
                robot_name, object_location = future.result()

                # Store result in a threadsafe manner
                with results_lock:
                    results.append((robot_name, object_location))

            finally:
                task_queue.task_done()

    def handle_a_robot(self, time_period=2, num_workers=3, total_iterations=5):
        """
        Manages sending multiple images for the single robot using two loops:
          1) A loop on the main thread that periodically enqueues tasks (futures).
          2) Multiple worker threads that pick tasks from the queue and wait on them.

        :param time_period: Seconds between each task being queued
        :param num_workers: Number of worker threads that handle completed futures
        :param total_iterations: How many total tasks (images) to capture and send
        """
        if not self.robot:
            self.logger.error("Robot not initialized. No tasks will be handled.")
            return

        if Debug_1:
            self.logger.info(
                f"Starting tasks for robot: {self.robot_name}\n"
                f"time_period = {time_period} seconds.\n"
                f"total_iterations = {total_iterations}.\n"
                f"num_workers = {num_workers}.\n"
            )

        # Prepare a queue of tasks (each will be a future returned by executor.submit)
        task_queue = Queue()
        results = []
        results_lock = threading.Lock()

        # Spawn worker threads that handle completed futures
        workers = []
        for _ in range(num_workers):
            t = threading.Thread(
                target=self._worker_loop, 
                args=(task_queue, results, results_lock),
                daemon=True
            )
            t.start()
            workers.append(t)

        # -------------------------------
        # Main loop to enqueue tasks
        # -------------------------------
        for iteration_index in range(total_iterations):
            if self.exit_signal.is_set():
                break

            # Capture an image from the robot
            image, timestamp = self.robot.robot.getImageTop()
            readable_date = datetime.fromtimestamp(timestamp).strftime('%d_%H_%M_%S_%f')
            image_file_name = f"{self.robot_name}_Top_{readable_date}_num_{iteration_index}"

            # Submit the detection call as a future
            future = self.executor.submit(
                self.send_image_async,
                self.robot_name,
                image,
                image_file_name
            )

            # Enqueue the future so that worker threads can handle completion
            task_queue.put(future)

            # Sleep the requested time period
            time.sleep(time_period)

        # Send sentinel None to stop workers
        for _ in range(num_workers):
            task_queue.put(None)

        # Wait for all tasks in the queue to complete
        task_queue.join()

        # Wait for all worker threads to exit
        for t in workers:
            t.join()

        # Finally, process results
        for (robot_name, object_location) in results:
            self.process_robot_tasks(robot_name, object_location)


def main():
    selected_robot_names = os.getenv('SELECTED_ROBOT_NAME', "").split(",")
    selected_robot_names = [name.strip() for name in selected_robot_names if name.strip()]

    try:
        num_of_requests = int(os.getenv('DATA_NUM', "1"))   # Default to 1 if not provided
        iterations = int(os.getenv('ITERATIONS', "1"))     # Default to 1 if not provided
        delay = int(os.getenv('DELAY', "1000")) / 1000.0   # Default to 1 second if not provided
        mode = os.getenv('MODE', "autonomous")
    except ValueError as e:
        raise ValueError("ITERATIONS and DELAY environment variables must be valid integers.") from e

    logger = setup_logger(f"{__name__}.main")
    logger.info(f"Starting with robot(s): {selected_robot_names}")

    # Initialize the manager with the first robot in the list (if any)
    manager = RobotGripperManager(selected_robot_names)

    # Handle Kubernetes SIGTERM and manual interrupt (SIGINT)
    signal.signal(signal.SIGTERM, manager.set_exit_signal)
    signal.signal(signal.SIGINT, manager.set_exit_signal)

    # Example calculation for time_period:
    # This is just an example formula. Adjust as desired.
    time_period = num_of_requests / delay

    # Run tasks for our single robot
    manager.handle_a_robot(
        time_period=time_period,
        num_workers=10,
        total_iterations=30000
    )

    logger.info("All tasks completed or exit signal received. Main thread is exiting.")


if __name__ == "__main__":
    main()
