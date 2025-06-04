import tempfile, os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Optional
from pydantic import HttpUrl
from PIL import Image, UnidentifiedImageError, ExifTags
import base64, io, torch
import cv2
import numpy as np
import time
from doctr.io import DocumentFile
from doctr.utils.geometry import rotate_image
from doctr.models import ocr_predictor

doctr_model = ocr_predictor(
    det_arch="db_resnet50",
    reco_arch="parseq",
    pretrained=True,
    det_bs=8,
    reco_bs=1024,
    assume_straight_pages=False,
    straighten_pages=True,
    detect_orientation=True,
).cuda().half()

# .5 Orientation Correction
def correct_orientation(image):
    doc = DocumentFile.from_images(image)
    result = doctr_model(doc)
    json_res = result.export()
    print("Orientation: ", json_res['pages'][0]['orientation']['value'])

    return rotate_image(cv2.imread(image), json_res['pages'][0]['orientation']['value'], expand=True)

# 1. Normalization
def normalize_image(image):
    norm_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    return cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)

# 4. Noise Removal
def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)

def increase_contrast_color(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge channels and convert back to BGR
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def preprocess_image_for_ocr(image_data):
    image = correct_orientation(image_data)

    if image is None:
        raise ValueError(f"Failed to load the image. Check the file path or format: {image_data}")

    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert image data to numpy array (if needed)
    # image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)

    # Resize to fit within a 2000x2000 box while maintaining aspect ratio
    height, width = image.shape[:2]
    scaling_factor = min(3072 / width, 3072 / height)
    if scaling_factor < 1:  # Only resize if the image is larger than 2000x2000
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Normalize the image
    image = normalize_image(image)

    # Noise removal
    image = remove_noise(image)

    # Increase contrast
    image = increase_contrast_color(image)

    cv2.imwrite('processed_image.png', image)
    return 'processed_image.png'

# default: Load the model on the available device(s)
model, processor = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-7B-Instruct",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("/models/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]


app = FastAPI()

class ImageContent(BaseModel):
    type: str = "image"
    image: str

class TextContent(BaseModel):
    type: str = "text"
    text: str

class Message(BaseModel):
    role: str
    content: List[Union[ImageContent, TextContent]]

class RequestBody(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 5000
    temperature: Optional[float] = 0.3
    repetition_penalty: Optional[float] = 1.2

@app.post("/v1/completions")
async def create_completion(request: RequestBody):

    messages = request.messages
    max_tokens = request.max_tokens
    temperature = request.temperature
    repetition_penalty = request.repetition_penalty

    # Temporarily save byte[] data to disk as a file and use its path
    temp_files = []
    try:
        messages_dict = []
        for message in messages:
            content_dict = []
            for content in message.content:
                if isinstance(content, ImageContent):
                    try:
                        # Decode the base64 image data
                        image_data = base64.b64decode(content.image)

                        # Open the image directly from bytes to check validity
                        print("Opening image...", flush=True)
                        Image.MAX_IMAGE_PIXELS = None   # disables the warning
                        image = Image.open(io.BytesIO(image_data))
                        print("Image opened", flush=True)
                        image.verify()  # This will raise an exception if the image is not valid
                        print("Image verified", flush=True)

                        # Reopen the image (since verify() can leave it in an unusable state)
                        image = Image.open(io.BytesIO(image_data))
                        # Determine the correct file extension based on image format
                        image_format = image.format.lower()
                        if image_format == "mpo":
                            image_format = "jpeg"

                        try:
                            for orientation in ExifTags.TAGS.keys():
                                if ExifTags.TAGS[orientation] == 'Orientation':
                                    break
                            exif = image._getexif()
                            if exif is not None and orientation in exif:
                                print(f"Orientation: {exif[orientation]}", flush=True)
                                if exif[orientation] == 3:
                                    image = image.rotate(180, expand=True)
                                elif exif[orientation] == 6:
                                    image = image.rotate(270, expand=True)
                                elif exif[orientation] == 8:
                                    image = image.rotate(90, expand=True)
                        except (AttributeError, KeyError, IndexError):
                            # Image has no EXIF orientation data
                            pass


                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_format}")
                        print(temp_file.name, flush=True)

                        # Save the image to a temporary file
                        image.save(temp_file, format=image_format.upper())
                        temp_file.flush()
                        temp_files.append(temp_file.name)

                        processed_file_name = preprocess_image_for_ocr(temp_file.name)

                        # Update the image field with the file path
                        content_dict.append({
                            "type": "image",
                            "image": f"file://{processed_file_name}"
                        })
                        print(content_dict, flush=True)
                    except UnidentifiedImageError as e:
                        return {"error": "Invalid image data: could not identify image."}
                    except Exception as e:
                        return {"error": str(e)}


                else:
                    # For TextContent, just append as is
                    content_dict.append(content.dict())

            # Add to the messages dictionary
            messages_dict.append({
                "role": message.role,
                "content": content_dict
            })

        print(messages_dict, flush=True)
        # Preparation for inference
        text = processor.apply_chat_template(
            messages_dict, tokenize=False, add_generation_prompt=True
        )
        print(f"text: {text}", flush=True)
        image_inputs, video_inputs = process_vision_info(messages_dict)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, repetition_penalty=repetition_penalty)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0], flush=True)

        return output_text[0]

    finally:
        # Clean up the temporary files
        for temp_file in temp_files:
            os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
