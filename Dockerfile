FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

LABEL maintainer="your.email@example.com"
LABEL description="YOLOv5 Model Export and Inference Environment"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/.cache

# Default command
CMD ["bash"]
