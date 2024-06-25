# 使用官方的NVIDIA CUDA基础镜像
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# 设置维护者信息
LABEL maintainer="z-xu@ist.osaka-u.ac.jp"

# 禁用交互模式
ARG DEBIAN_FRONTEND=noninteractive

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
    && rm -rf /var/lib/apt/lists/*

# 设置Python3.10为python3默认版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 设置Python3.10为python命令对象
RUN ln -s /usr/bin/python3.10 /usr/local/bin/python

# 安装pip
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.10 get-pip.py && rm get-pip.py

# 安装PyTorch（包括CUDA和cuDNN支持）
RUN python -m pip --no-cache-dir install --upgrade \
        --pre torch==2.0.0 torchaudio==2.0.1 torchvision==0.15.1 -f \
        https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html

# 安装其他常用的Python包
RUN pip install --upgrade \
    datasets \
    jupyterlab \
    matplotlib \
    numpy \
    pandas

# 暴露需要的端口（如果有）
# EXPOSE 8888

# 如果你使用Jupyter Notebook，可以使用以下命令启动Jupyter
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
