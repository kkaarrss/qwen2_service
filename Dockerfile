# Stage 1: Builder
# Use a slightly more comprehensive base image if needed for building,
# but stick to the same CUDA version for consistency if possible.
# The runtime image will be based on the -runtime variant.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder
# Using -devel here gives us build tools, but we'll switch to -runtime for the final image.

# Set environment variables for NVIDIA (can be set in final stage too)
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install system dependencies needed for building Python packages
# (e.g., gcc, g++, make, etc., if any of your pip packages compile C extensions)
# For now, keeping it minimal as PyTorch often comes pre-compiled.
# We still need python3-pip and python3-venv for the venv.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment using the system python3
RUN python3 -m venv --copies /opt/venv
# Using --copies instead of default symlinks might make the venv more portable when copied.
# This copies the Python interpreter binary into the venv instead of symlinking.
# This can make the venv larger but more self-contained.


# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Activate and install (PATH is already set)
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final Runtime Image ---
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install only essential runtime system dependencies
# libgl1, libglib2.0-0 are often needed for OpenCV/PyTorch GUIs or some operations.
# If your app is truly headless and doesn't need them, you might try removing them.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code and necessary utils
COPY text_unsloth_2_5.py .

# Expose the FastAPI port (ensure this matches your uvicorn command)
EXPOSE 8001

# User to run the application (optional, but good practice for security)
# RUN groupadd --gid 1000 appuser && \
#     useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser
# WORKDIR /home/appuser/app # Change workdir if using non-root user
# COPY --from=builder --chown=appuser:appuser /app/qwen_vl_utils.py .
# COPY --chown=appuser:appuser text_unsloth_2_5.py .
# USER appuser

# Command to run your FastAPI application
# Ensure "text_unsloth_2_5:app" matches your filename and FastAPI app instance name
CMD ["uvicorn", "text_unsloth_2_5:app", "--host", "0.0.0.0", "--port", "8001"]
