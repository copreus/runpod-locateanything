# Use RunPod's official PyTorch base image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install the requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the python script
COPY handler.py .

# Start the RunPod API handler
CMD [ "python", "-u", "/app/handler.py" ]
