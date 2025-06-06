import tempfile
import os
import base64
import io
import torch

import time
import uuid
import requests # For downloading images from URLs
import asyncio # For streaming
import json # For streaming JSON objects

from PIL import Image, UnidentifiedImageError
from contextlib import asynccontextmanager # Import for lifespan manager

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers.generation.streamers import TextIteratorStreamer
from threading import Thread

from qwen_vl_utils import process_vision_info # Ensure this is the official version

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Union, Optional, Dict, Literal, AsyncGenerator

# --- Global Variables for Model and Processor ---
model = None
processor = None

# --- Configuration ---
MODEL_NAME = "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"
MIN_PIXELS = 256*28*28
MAX_IMAGE_PIXELS = 400000000

# --- Lifespan Manager for Model Loading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Loads the model and processor on startup and cleans up on shutdown.
    This replaces the deprecated `on_event("startup")` decorator.
    """
    global model, processor

    print(f"INFO: Loading Qwen VL model ({MODEL_NAME}) on startup...")
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            trust_remote_code=True
        )
        print("INFO: Qwen VL model and processor loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load Qwen VL model or processor during startup: {e}")
        import traceback
        traceback.print_exc()
        # In a production environment, you might want the app to exit if model loading fails.

    yield

    # Cleanup on shutdown
    print("INFO: Application shutting down. Releasing resources.")
    model = None
    processor = None
    torch.cuda.empty_cache()


# --- FastAPI App ---
# Use the lifespan manager to handle startup and shutdown logic
app = FastAPI(title="Qwen VL OpenAI-Compatible API (Cleaned)", lifespan=lifespan)


# --- Simplified Image Handling ---
def save_image_to_temp_file(image_bytes: bytes, image_format: str, output_dir: str) -> str:
    """Saves image bytes to a temporary file and returns its path."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_image:
            pil_image.verify()

        safe_suffix = f".{image_format.lower().split('/')[-1]}"
        if safe_suffix == ".jpg": safe_suffix = ".jpeg"

        fd, temp_image_path = tempfile.mkstemp(suffix=safe_suffix, dir=output_dir)
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(image_bytes)
        return temp_image_path
    except Exception as e:
        print(f"ERROR: Failed to save image to temp file: {e}")
        raise

# --- Pydantic Models ---
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "custom"
    # New field to indicate model capabilities, including vision
    capabilities: Optional[Dict[str, bool]] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

class OpenAIImageURL(BaseModel):
    url: Union[HttpUrl, str]
    detail: Optional[str] = "auto"

class OpenAIContentPartText(BaseModel):
    type: Literal["text"]
    text: str

class OpenAIContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: OpenAIImageURL

OpenAIContentPart = Union[OpenAIContentPartText, OpenAIContentPartImage]

class OpenAIMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[OpenAIContentPart]]
    name: Optional[str] = None

class OpenAIChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL_NAME
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = 1536
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    repetition_penalty: Optional[float] = 1.1

class StreamChoiceDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[Literal["assistant"]] = None

class StreamChoice(BaseModel):
    index: int
    delta: StreamChoiceDelta
    finish_reason: Optional[Literal["stop", "length"]] = None

class OpenAIChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]

class ResponseMessage(BaseModel):
    role: Literal["assistant"]
    content: str

class ChatCompletionChoice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: Optional[Literal["stop", "length"]] = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageInfo] = Field(default_factory=UsageInfo)


# --- API Endpoints ---
@app.get("/v1/models", response_model=ModelList)
async def list_models_endpoint():
    """Lists the available models, indicating vision capabilities."""
    global MODEL_NAME
    model_card = ModelCard(
        id=MODEL_NAME,
        capabilities={"vision": True}  # Explicitly state vision capability
    )
    return ModelList(data=[model_card])

async def true_generate_response_stream(
    inputs_on_device: Dict,
    gen_kwargs: Dict,
    request_id: str,
    model_name_str: str,
    hf_processor # Passed from global processor
) -> AsyncGenerator[str, None]:
    streamer = TextIteratorStreamer(hf_processor, skip_prompt=True, skip_special_tokens=True)
    generation_thread_kwargs = {**inputs_on_device, **gen_kwargs, "streamer": streamer}
    thread = Thread(target=model.generate, kwargs=generation_thread_kwargs)
    thread.start()
    assistant_role_sent = False
    for new_text in streamer:
        if new_text:
            delta_content_dict = {}
            if not assistant_role_sent:
                delta_content_dict['role'] = "assistant"
                assistant_role_sent = True
            delta_content_dict['content'] = new_text

            delta = StreamChoiceDelta(**delta_content_dict)
            stream_response = OpenAIChatCompletionStreamResponse(
                id=request_id, model=model_name_str, choices=[StreamChoice(index=0, delta=delta)]
            )
            yield f"data: {stream_response.model_dump_json()}\n\n"
            await asyncio.sleep(0.001) # Minimal sleep to allow I/O

    thread.join()
    final_delta = StreamChoiceDelta()
    final_choice = StreamChoice(index=0, delta=final_delta, finish_reason="stop")
    final_stream_response = OpenAIChatCompletionStreamResponse(
        id=request_id, model=model_name_str, choices=[final_choice]
    )
    yield f"data: {final_stream_response.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def create_chat_completion(request: OpenAIChatCompletionRequest, raw_http_request: Request):
    global model, processor, MODEL_NAME

    if model is None or processor is None:
        print("ERROR: Model or processor not loaded during request. Check startup logs.")
        raise HTTPException(status_code=503, detail="Model is not available. Please try again later.")

    # Optional: Log basic request info, less verbose than full body
    print(f"INFO: Received request for model '{request.model}', stream: {request.stream}, messages: {len(request.messages)}")

    if request.n is not None and request.n > 1:
        raise HTTPException(status_code=400, detail="Generating multiple choices (n > 1) is not supported.")

    temp_files_to_clean = []
    temp_dir_prefix = "qwen_vl_api_tmp_"
    temp_dir = tempfile.mkdtemp(prefix=temp_dir_prefix)
    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    try:
        qwen_messages_dict = []

        for oai_message in request.messages:
            qwen_content_parts = []
            if isinstance(oai_message.content, str):
                qwen_content_parts.append({"type": "text", "text": oai_message.content})
            elif isinstance(oai_message.content, list):
                for part in oai_message.content:
                    if part.type == "text":
                        qwen_content_parts.append({"type": "text", "text": part.text})
                    elif part.type == "image_url":
                        image_url_data = str(part.image_url.url)
                        image_bytes, image_format = None, "png"
                        if image_url_data.startswith("http"):
                            try:
                                img_response = requests.get(image_url_data, timeout=20)
                                img_response.raise_for_status()
                                image_bytes = img_response.content
                                content_type = img_response.headers.get("Content-Type", "").lower()
                                if "jpeg" in content_type or "jpg" in content_type: image_format = "jpeg"
                                elif "png" in content_type: image_format = "png"
                                elif "webp" in content_type: image_format = "webp"
                            except requests.RequestException as e:
                                print(f"ERROR: Failed to download image from URL {image_url_data}: {e}")
                                raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")
                        elif image_url_data.startswith("data:image"):
                            try:
                                header, encoded = image_url_data.split(",", 1)
                                image_format = header.split("/")[1].split(";")[0].lower()
                                if image_format not in ["jpeg", "jpg", "png", "webp", "gif", "bmp"]:
                                    raise ValueError(f"Unsupported image format in data URI: {image_format}")
                                image_bytes = base64.b64decode(encoded)
                            except Exception as e:
                                print(f"ERROR: Invalid base64 image data: {e}")
                                raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")
                        else:
                            print(f"ERROR: Unsupported image URL format: {image_url_data}")
                            raise HTTPException(status_code=400, detail="Unsupported image URL format.")

                        if image_bytes:
                            try:
                                # Save to temp file for process_vision_info
                                temp_image_path = save_image_to_temp_file(image_bytes, image_format, temp_dir)
                                temp_files_to_clean.append(temp_image_path)
                                qwen_content_parts.append({"type": "image", "image": f"file://{temp_image_path}"})
                            except UnidentifiedImageError as e_unid:
                                print(f"ERROR: Invalid or unidentified image: {e_unid}")
                                raise HTTPException(status_code=400, detail=f"Invalid or unidentified image: {e_unid}")
                            except Exception as e_pil:
                                print(f"ERROR: Error processing image with PIL/saving: {e_pil}")
                                raise HTTPException(status_code=500, detail=f"Error processing image: {e_pil}")
            qwen_messages_dict.append({"role": oai_message.role, "content": qwen_content_parts})

        text_prompt = processor.apply_chat_template(qwen_messages_dict, tokenize=False, add_generation_prompt=True)

        final_image_inputs_for_processor, final_video_inputs_for_processor = [], []
        contains_qwen_image_parts = any("image" == part.get("type") for m in qwen_messages_dict for part in m.get("content", []) if isinstance(m.get("content"), list))

        if contains_qwen_image_parts:
            try:
                img_proc_arg = processor.image_processor if hasattr(processor, 'image_processor') else None
                pvi_pil_images, pvi_video_paths, _pvi_video_info = process_vision_info(qwen_messages_dict, img_proc_arg)
                final_image_inputs_for_processor = pvi_pil_images
                final_video_inputs_for_processor = pvi_video_paths
                if final_image_inputs_for_processor:
                    print(f"INFO: process_vision_info returned {len(final_image_inputs_for_processor)} PIL image(s).")
            except Exception as e_pvi:
                print(f"ERROR: During process_vision_info: {type(e_pvi).__name__} - {e_pvi}. Vision inputs will be empty.")
                # Fallback handled by final_image_inputs_for_processor remaining empty

        inputs = processor(
            text=[text_prompt],
            images=final_image_inputs_for_processor if final_image_inputs_for_processor else None,
            videos=final_video_inputs_for_processor if final_video_inputs_for_processor else None,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
            print(f"INFO: Image successfully processed into pixel_values tensor shape: {inputs['pixel_values'].shape}")
        elif contains_qwen_image_parts and (not final_image_inputs_for_processor):
            print(f"WARNING: Request contained image parts, but no images were passed to the model (pixel_values is None).")

        gen_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "top_p": request.top_p if request.top_p is not None else 1.0,
            "repetition_penalty": request.repetition_penalty if request.repetition_penalty is not None else 1.1,
        }

        if request.stream:
            print(f"INFO: Request {request_id} - Streaming response.")
            return StreamingResponse(
                true_generate_response_stream(inputs, gen_kwargs, request_id, MODEL_NAME, processor),
                media_type="text/event-stream"
            )
        else:
            print(f"INFO: Request {request_id} - Non-streaming response.")
            with torch.no_grad():
                generated_ids = model.generate(**inputs, **gen_kwargs)

            prompt_tokens_count = len(inputs["input_ids"][0]) if "input_ids" in inputs else 0
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            completion_tokens_count = len(generated_ids_trimmed[0]) if generated_ids_trimmed else 0
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            usage = UsageInfo(
                prompt_tokens=prompt_tokens_count,
                completion_tokens=completion_tokens_count,
                total_tokens=prompt_tokens_count + completion_tokens_count
            )
            response_message = ResponseMessage(role="assistant", content=output_text)
            choice = ChatCompletionChoice(index=0, message=response_message, finish_reason="stop")
            return OpenAIChatCompletionResponse(
                id=request_id, model=MODEL_NAME, choices=[choice], usage=usage
            )

    except HTTPException as http_exc:
        print(f"ERROR: Request {request_id} - HTTPException: Status {http_exc.status_code}, Detail: {http_exc.detail}")
        raise
    except Exception as e:
        print(f"CRITICAL ERROR: Request {request_id} - Unexpected error: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        error_detail_str = str(e) if e is not None else "Unknown internal server error."
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_detail_str}")
    finally:
        # print(f"DEBUG: Cleaning up temporary directory: {temp_dir}") # Can be noisy
        for f_path in temp_files_to_clean:
            try:
                if os.path.exists(f_path): os.remove(f_path)
            except Exception as e_clean: print(f"  Warning: Error cleaning temp file {f_path}: {e_clean}")
        try:
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
        except Exception as e_clean_dir: print(f"  Warning: Error cleaning temp dir {temp_dir}: {e_clean_dir}")
        print(f"INFO: Request {request_id} - Processing finished.")

# --- Main execution ---
if __name__ == "__main__":
    import uvicorn
    # Correctly get the current file's name for uvicorn's reload feature
    current_file_name = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{current_file_name}:app", host="0.0.0.0", port=31000, reload=True, log_level="info")
