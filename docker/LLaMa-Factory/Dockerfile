# 使用官方的NVIDIA CUDA基础镜像
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# 设置维护者信息
LABEL maintainer="z-xu@ist.osaka-u.ac.jp"

# 禁用交互模式
ARG DEBIAN_FRONTEND=noninteractive

# Define installation arguments
ARG INSTALL_BNB=false
ARG INSTALL_VLLM=false
ARG INSTALL_DEEPSPEED=false
ARG PIP_INDEX=https://pypi.org/simple

# Set the working directory
WORKDIR /app

# 添加python3.10软件源
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa

# 更新包列表并安装常用工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    wget \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pkg-resources \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# 设置Python3.10为python3默认版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 设置Python3.10为python命令对象
RUN ln -s /usr/bin/python3.10 /usr/local/bin/python

# 安装pip
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.10 get-pip.py && rm get-pip.py

# 安装PyTorch（包括CUDA和cuDNN支持）
RUN python -m pip --no-cache-dir install --upgrade \
        --pre torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 -f \
        https://download.pytorch.org/whl/cu118

# Install the requirements
COPY requirements.txt /app/
RUN pip config set global.index-url $PIP_INDEX
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# 安装其他常用的Python包
RUN pip install --upgrade \
    datasets \
    flash-attn \
    pytest \
    setuptools==69.5.1

# Copy the rest of the application into the image
COPY . /app/

# Install the LLaMA Factory
RUN EXTRA_PACKAGES="metrics"; \
    if [ "$INSTALL_BNB" = "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},bitsandbytes"; \
    fi; \
    if [ "$INSTALL_VLLM" = "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},vllm"; \
    fi; \
    if [ "$INSTALL_DEEPSPEED" = "true" ]; then \
        EXTRA_PACKAGES="${EXTRA_PACKAGES},deepspeed"; \
    fi; \
    pip install -e .[$EXTRA_PACKAGES] && \
    pip uninstall -y transformer-engine
