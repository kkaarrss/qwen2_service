# Use NVIDIA's official CUDA base image with PyTorch preinstalled
#FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Set environment variables for NVIDIA
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Set working directory inside the container
WORKDIR /app

# Set noninteractive mode for apt-get to avoid tzdata prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    libglx-mesa0 \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate the virtual environment and install dependencies
#RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
RUN /opt/venv/bin/pip install -r requirements.txt

COPY text_unsloth_2_5.py .

# Make the virtual environment's Python the default for subsequent RUN commands
ENV PATH="/opt/venv/bin:$PATH"

# Expose the FastAPI port
EXPOSE 31000

# Command to run your FastAPI application
CMD ["uvicorn", "text_unsloth_2_5:app", "--host", "0.0.0.0", "--port", "31000"]
