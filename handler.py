import runpod
import torch
from PIL import Image
import base64
import io
from transformers import AutoProcessor, AutoModel

model_id = "nvidia/LocateAnything-3B"

# 1. Load the model into the GPU when the container boots up
print("Downloading and loading LocateAnything-3B...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
).to("cuda").eval()
print("Model loaded successfully!")

def handler(event):
    # 2. Extract the image and prompt you send via the API
    job_input = event.get('input', {})
    image_b64 = job_input.get('image')
    prompt = job_input.get('prompt', "Find all objects") # Default prompt if you don't provide one
    
    if not image_b64:
        return {"error": "No image provided in the API request"}
    
    # 3. Decode the base64 image
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 4. Run the model and return the bounding boxes
    with torch.no_grad():
        output = model.predict(image, prompt)
        
    return {"bounding_boxes": output}

# Start the Serverless API
runpod.serverless.start({"handler": handler})
