# Use the NVIDIA CUDA runtime base image with CUDA 12.1 and cuDNN support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-setuptools \
    tmux \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    wget \
    unzip \
    libglew-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip3 install --no-cache-dir torch==2.0.1+cu121 torchvision==0.15.2+cu121 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install MuJoCo 3.2.3
RUN wget https://mujoco.org/download/mujoco-linux-x86_64-3.2.3.tar.gz && \
    mkdir -p /root/.mujoco && \
    tar -zxvf mujoco-linux-x86_64-3.2.3.tar.gz -C /root/.mujoco && \
    rm mujoco-linux-x86_64-3.2.3.tar.gz

# Set environment variables for MuJoCo
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco-3.2.3/bin:${LD_LIBRARY_PATH}
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco-3.2.3
ENV MUJOCO_GL=egl

# Install dm_control
RUN pip3 install --no-cache-dir dm_control

# Copy project files
COPY . /app

# Set the working directory
WORKDIR /app

# Default command to run the training task
CMD ["python3", "-O", "main.py", "train", "--cfg=tasks/defaults/sac.yml"]
