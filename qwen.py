import tempfile, os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Optional
from pydantic import HttpUrl
from PIL import Image, UnidentifiedImageError
import base64, io

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/models/Qwen2-VL-7B-Instruct", torch_dtype="float16", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
#processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

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
    max_tokens: Optional[int] = 50

@app.post("/v1/completions")
async def create_completion(request: RequestBody):

    messages = request.messages
    max_tokens = request.max_tokens

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
                        image = Image.open(io.BytesIO(image_data))
                        image.verify()  # This will raise an exception if the image is not valid

                        # Reopen the image (since verify() can leave it in an unusable state)
                        image = Image.open(io.BytesIO(image_data))

                        # Determine the correct file extension based on image format
                        image_format = image.format.lower()
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_format}")
                        
                        # Save the image to a temporary file
                        image.save(temp_file, format=image_format.upper())
                        temp_file.flush()
                        temp_files.append(temp_file.name)

                        # Update the image field with the file path
                        content_dict.append({
                            "type": "image",
                            "image": f"file://{temp_file.name}"
                        })
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

        print(messages_dict)
        # Preparation for inference
        text = processor.apply_chat_template(
            messages_dict, tokenize=False, add_generation_prompt=True
        )
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
        generated_ids = model.generate(**inputs, max_new_tokens=10000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0])

        return output_text[0]

    finally:
        # Clean up the temporary files
        for temp_file in temp_files:
           os.remove(temp_file)
#           print(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
