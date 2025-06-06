# Qwen-VL OpenAI-Compatible API Server (qwen2_service)

This project provides an OpenAI-compatible API endpoint for interacting with a Quantized Qwen2.5-VL model (specifically `unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit` by default), served via FastAPI. It allows you to send text and image inputs and receive text-based responses, mimicking the OpenAI Chat Completions API structure.

**Repository:** [https://github.com/kkaarrss/qwen2_service](https://github.com/kkaarrss/qwen2_service)

## Features

-   **OpenAI Compatibility:** Exposes `/v1/chat/completions` and `/v1/models` endpoints.
-   **Vision-Language Model:** Powered by Qwen2.5-VL, capable of understanding both text and images.
-   **4-bit Quantization:** Uses `BitsAndBytesConfig` for efficient model loading and reduced memory footprint.
-   **Streaming Support:** Provides real-time, token-by-token streaming for responses.
-   **Flexible Image Handling:** Accepts images as `https://` URLs or `data:` URI (base64) strings.
-   **Dockerized:** Includes a Dockerfile and a pre-built image on Docker Hub (`pluskars/qwen-vl`) for easy deployment.

## Prerequisites

### Common for All Methods:

-   **NVIDIA GPU & Drivers:** Required for running the model efficiently. Ensure you have compatible NVIDIA drivers installed.
-   **CUDA Toolkit:** The version should be compatible with the PyTorch version used. The Docker image `pluskars/qwen-vl` is built with a CUDA 12.8.1 base.
-   **Python (for local setup):** Python 3.10 or newer is recommended.

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
    This command runs the container in detached mode (`-d`), maps port `31000` to the host, enables GPU access, and mounts your Hugging Face cache to speed up model downloads on subsequent runs.

    ```bash
    # Ensure NVIDIA Container Toolkit is installed and Docker daemon is restarted if needed.
    # Create the cache directory on your host if it doesn't exist:
    mkdir -p ~/.cache/huggingface

    docker run -d \
        --gpus all \
        -p 31000:31000 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --name qwen-vl-api \
        pluskars/qwen-vl:latest
    ```

3.  **Check container logs (optional):**
    Wait a few minutes for the model to load, then check the logs to confirm it's running.
    ```bash
    docker logs qwen-vl-api -f
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
    (Create the `requirements.txt` file as specified below)
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the FastAPI application:**
    ```bash
    python text_unsloth_2_5.py
    ```
    The server will start on `http://0.0.0.0:31000`.

### `requirements.txt` Content

Create a `requirements.txt` file with these exact versions for reproducibility:
```txt
fastapi
uvicorn
pydantic
transformers==4.41.2
torch==2.3.0
torchvision
Pillow
requests
bitsandbytes
accelerate
qwen-vl-utils==0.0.11
```

## API Endpoints

### 1. List Models

-   **Endpoint:** `GET /v1/models`
-   **Description:** Returns a list of available models and their capabilities.
-   **Example Response:**
    ```json
    {
      "object": "list",
      "data": [
        {
          "id": "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
          "object": "model",
          "created": 1721921387,
          "owned_by": "custom",
          "capabilities": {
            "vision": true
          }
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
                "url": "data:image/jpeg;base64,/9j/4AAQSk...=="
              }
            }
          ]
        }
      ],
      "max_tokens": 150,
      "stream": false
    }
    ```

## Using with a Web UI (Open WebUI)

You can connect this API to a user-friendly chat interface like [**Open WebUI**](https://github.com/open-webui/open-webui) to get a ChatGPT-like experience with image uploads, all running locally.

### Step 1: Run the Qwen-VL API Server

Follow the "Running with Docker" instructions above to start the API server.

### Step 2: Run Open WebUI

Run the Open WebUI container using Docker. The `--add-host` flag is crucial as it allows the Open WebUI container to communicate with the Qwen-VL API container.

```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui --restart always \
  ghcr.io/open-webui/open-webui:main
```

### Step 3: Configure Open WebUI

1.  Open your browser and navigate to `http://localhost:3000`.
2.  Create your admin account on the first launch.
3.  Click the settings gear icon ⚙️ in the top right, then go to **Connections**.
4.  Set the following values to connect to your local API server:
    -   **Connection URL**: `http://host.docker.internal:31000/v1`
    -   **API Key**: `1234` (The server doesn't require a key, but the UI needs a placeholder value).
5.  Click **Save**. Open WebUI will connect to your server and automatically pull the model list.

### Step 4: Start Chatting

1.  Go back to the main chat screen.
2.  At the top, click **"Select a Model"**.
3.  You should see your model listed: `unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit`. Select it.
4.  You can now start a conversation! Use the paperclip icon 📎 to upload images.

## Example `curl` Requests

**List Models:**
```bash
curl http://localhost:31000/v1/models
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

**Chat Completion (Streaming):**
```bash
curl -N -X POST http://localhost:31000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Accept: text/event-stream" \
-d '{
  "messages": [{"role": "user", "content": "Tell me a short story about a robot who discovers music."}],
  "stream": true,
  "max_tokens": 150
}'
```

## Troubleshooting

-   **GPU Not Detected in Docker:** Ensure the NVIDIA Container Toolkit is correctly installed and your Docker daemon has been restarted. Test with `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`.
-   **Port Conflicts:** If port `31000` or `3000` is in use, the `docker run` command will fail. Free up the port or change the mapping (e.g., `-p 31001:31000`).
-   **Open WebUI Can't Connect:**
    -   Verify the API server is running with `docker logs qwen-vl-api`.
    -   Ensure you used the correct URL: `http://host.docker.internal:31000/v1`. `localhost` will not work from inside the Open WebUI container.
