# 使用官方的NVIDIA CUDA基础镜像
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# 设置维护者信息
LABEL maintainer="z-xu@ist.osaka-u.ac.jp"

# 禁用交互模式
ARG DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /app

# 更新包列表并安装常用工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    wget \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pkg-resources \
    python3-setuptools \
    # 设置Python3.8为python3默认版本
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    # 设置Python3.8为python命令对象
    && ln -s /usr/bin/python3.8 /usr/local/bin/python \
    # 安装pip
    && wget https://bootstrap.pypa.io/get-pip.py && python3.8 get-pip.py && rm get-pip.py \
    # 删除apt缓存
    && rm -rf /var/lib/apt/lists/*

# 安装PyTorch、常用Python依赖，并删除缓存
COPY requirements.txt /app/
RUN python -m pip --no-cache-dir install --upgrade \
        --pre torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -f \
        https://download.pytorch.org/whl/cu118 \
    && python -m pip install -r requirements.txt \
    && rm -rf ~/.cache/pip