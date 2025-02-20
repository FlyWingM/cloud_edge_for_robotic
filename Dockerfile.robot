### Dockerfile.robot

# Base image
FROM python:3.9-slim

# Install system dependencies in a single RUN command to reduce layers
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    cmake \
    build-essential \
    patchelf \
    pkg-config \
    libhdf5-dev \
    libhdf5-serial-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to avoid strict metadata checks, and install critical Python dependencies
RUN pip install --upgrade pip==23.0.1 && \
    pip install numpy Cython

# Set working directory
WORKDIR /app

# Add a non-root user
RUN useradd -m robot_user
USER robot_user

# Copy and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application files and directories
COPY robot.py /app/
COPY config/ /app/config/
COPY utils/ /app/utils/
COPY grasping/ /app/grasping/
COPY images/ /app/images/


# Expose port 5000 (if needed)
EXPOSE 5000

# Command to run the PhysicalRobotGripper service
CMD ["python", "/app/robot.py"]
