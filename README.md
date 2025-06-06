# Qwen-VL OpenAI-Compatible API Server (qwen2_service)

This project provides an OpenAI-compatible API endpoint for interacting with a Quantized Qwen2.5-VL model (specifically `unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit` by default), served via FastAPI. It allows you to send text and image inputs and receive text-based responses, mimicking the OpenAI Chat Completions API structure.

**Repository:** [https://github.com/kkaarrss/qwen2_service](https://github.com/kkaarrss/qwen2_service)

## Features

-   **OpenAI Compatibility:** Exposes `/v1/chat/completions` and `/v1/models` endpoints.
-   **Vision-Language Model:** Powered by Qwen2.5-VL.
-   **4-bit Quantization:** Uses `BitsAndBytesConfig` for efficient model loading.
-   **Streaming Support:** Provides real token-by-token streaming for responses.
-   **Image Handling:** Accepts images as URLs or base64 strings.
-   **Simplified Image Preprocessing:** Relies on PIL for basic validation and the Hugging Face `AutoProcessor` for model-specific image transformations.
-   **Dockerized:** Includes a Dockerfile and a pre-built image on Docker Hub (`pluskars/qwen-vl`).

## Prerequisites

### Common for All Methods:

-   **NVIDIA GPU & Drivers:** Required for running the model efficiently. Ensure you have compatible NVIDIA drivers installed.
-   **CUDA Toolkit:** The version should be compatible with the PyTorch version used (e.g., CUDA 11.8 or 12.x). The Docker image `pluskars/qwen-vl` is built with a CUDA 12.8.1 base.
-   **Python (for local setup):** Python 3.10 or newer is recommended.

### For Running Without Docker (Local Setup from Source):

-   **Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
-   **Python Dependencies:** Install using the `requirements.txt` file.

### For Running With Docker:

-   **Docker:** Install Docker Engine.
-   **NVIDIA Container Toolkit:** Essential for GPU access within Docker containers. Follow the installation guide for your OS: [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Setup & Running

The main application script is `text_unsloth_2_5.py`.

### Option 1: Running with Docker (Recommended - Easiest)

This uses the pre-built image from Docker Hub.

1.  **Pull the Docker image:**
    ```bash
    docker pull pluskars/qwen-vl:latest
    ```

2.  **Run the Docker container:**
    This command runs the container in detached mode (`-d`), maps port `31000` from the container to the host (the application inside listens on port `31000`), enables GPU access, and mounts your local Hugging Face cache to speed up model downloads on subsequent runs if models weren't baked into the image or if you switch models.

    ```bash
    # Ensure NVIDIA Container Toolkit is installed and Docker daemon is restarted if needed.
    # Create the cache directory on your host if it doesn't exist:
    mkdir -p ~/.cache/huggingface

    docker run -d \
        --gpus all \
        -p 31000:31000 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        pluskars/qwen-vl:latest
    ```

3.  **Check container logs (optional):**
    Find the container ID using `docker ps`.
    ```bash
    docker logs <container_id_or_name> -f
    ```
    The server will be available at `http://localhost:31000`.

### Option 2: Running Locally from Source (Without Docker)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kkaarrss/qwen2_service.git
    cd qwen2_service
    ```

2.  **Create and activate a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows
    ```

3.  **Install dependencies:**
    (Ensure you have a `requirements.txt` file as specified below)
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the FastAPI application using Uvicorn:**
    The script `text_unsloth_2_5.py` contains the Uvicorn runner in its `if __name__ == "__main__":` block.
    ```bash
    python text_unsloth_2_5.py
    ```
    The server will start on `http://0.0.0.0:31000` (or the port configured in `text_unsloth_2_5.py`).

### `requirements.txt` Content

If setting up locally, create a `requirements.txt` file with:
```txt
bitsandbytes==0.45.3
fastapi
pydantic
transformers==4.51.3
torch==2.6.0
Pillow
qwen-vl-utils==0.0.11
uvicorn
torchvision==0.21.0
accelerate==0.26.1

```

## API Endpoints

The server exposes the following OpenAI-compatible endpoints:

### 1. List Models

-   **Endpoint:** `GET /v1/models`
-   **Description:** Returns a list of available models.
-   **Example Response:**
    ```json
    {
      "object": "list",
      "data": [
        {
          "id": "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
          "object": "model",
          "created": 1677610600,
          "owned_by": "custom"
        }
      ]
    }
    ```

### 2. Chat Completions

-   **Endpoint:** `POST /v1/chat/completions`
-   **Description:** Generates a model response. Supports text and image inputs, and streaming.
-   **Request Body (Example with Image):**
    ```json
    {
      "model": "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "What is in this image?"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": "data:image/jpeg;base64,/9j/4AAQSk...==" // Or an https:// URL
              }
            }
          ]
        }
      ],
      "max_tokens": 150,
      "stream": false, // Set to true for streaming
      "temperature": 0.7
    }
    ```
-   **Non-Streaming Response & Streaming Response:** (Examples as in the previous README version)

*(Keep the Non-Streaming and Streaming Response examples from the previous README here)*

## Example `curl` Requests

*(Keep the curl examples from the previous README here, ensuring the port is 31000)*

**List Models:**
```bash
curl http://localhost:31000/v1/models
```

**Chat Completion (Non-Streaming, Text-Only):**
```bash
curl -X POST http://localhost:31000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "messages": [{"role": "user", "content": "Hello, how are you?"}],
  "max_tokens": 50
}'
```

**Chat Completion (Non-Streaming, with Image URL):**
```bash
curl -X POST http://localhost:31000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image:"},
        {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}}
      ]
    }
  ],
  "max_tokens": 100
}'
```

**Chat Completion (Streaming, with Base64 Image):**
```bash
# Replace YOUR_BASE64_IMAGE_STRING with actual base64 data
BASE64_IMAGE="YOUR_BASE64_IMAGE_STRING"

curl -N -X POST http://localhost:31000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Accept: text/event-stream" \
-d @- <<EOF
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this picture?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${BASE64_IMAGE}"}}
      ]
    }
  ],
  "stream": true,
  "max_tokens": 100
}
EOF
```
*Note: Added `-N` (no-buffering) to the streaming `curl` example for better SSE display.*

## Troubleshooting

-   **GPU Not Detected in Docker:** Ensure NVIDIA Container Toolkit is correctly installed and your Docker daemon is configured/restarted. Test with `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`.
-   **Model Download Issues:** Ensure network connectivity. If using Docker, mounting `~/.cache/huggingface` can help persist downloads if the model isn't fully baked into the `pluskars/qwen-vl` image or for future model updates.
-   **Port Conflicts:** If port `31000` is already in use, the Docker command will fail to map it. Ensure the port is free or change the mapping (e.g., `-p 31001:31000`).
-   **`qwen-vl-utils` Version:** Ensure the pip-installed version is compatible with your `transformers` library version.

## Building the Docker Image (Optional - if modifying the source)

If you clone the repository and make changes to `text_unsloth_2_5.py` or `Dockerfile`, you can build your own image:
```bash
# In the root of the cloned repository (qwen2_service)
docker build -t my-qwen-vl-api:latest .
```
Then run your custom image instead of `pluskars/qwen-vl:latest`.
Make sure your `Dockerfile` copies `text_unsloth_2_5.py` and installs `qwen-vl-utils` from `requirements.txt`.
```
