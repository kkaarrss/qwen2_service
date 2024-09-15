At least install fastapi uvicorn Pillow. Maybe more but I forgot what. Look at the error messages what is missing and install missing modules accordingly.

run:
pip install transformers fastapi uvicorn Pillow --upgrade



run the service with:
uvicorn qwen:app --host 0.0.0.0 --port 8000



To try it out with curl:
convert image to base64:

base64 -i "IMAGE NAME HERE.jpg" -o test_base64.txt



Then run the curl command:

curl -X POST "http://localhost:8000/v1/completions" -H "Content-Type: application/json" -d '{ "messages": [ { "role": "user", "content": [ { "type": "image", "image": "'$(cat test_base64.txt)'" }, { "type": "text", "text": "Output the handwritten text from the image." } ] } ], "max_tokens": 50 }'
