# Memo for LLaMa-Factory

## Docker Image

```Bash
docker build -f $(pwd)/Dockerfile --build-arg INSTALL_BNB=false --build-arg INSTALL_VLLM=false --build-arg INSTALL_DEEPSPEED=false --build-arg PIP_INDEX=https://pypi.org/simple -t llamafactory:latest .
```

## Docker Container

```Bash
docker run -it --gpus=all -v $(pwd)/hf_cache:/root/.cache/huggingface/ -v $(pwd)/data:/app/data  -v $(pwd)/output:/app/output -p 7860:7860 -p 8000:8000 --shm-size 16G --name llamafactory llamafactory:latest
```

## LLaMa-Factory Command

```Bash
llamafactory-cli webui
```

## Python3.10

```Bash
apt-get update && apt-get install -y --no-install-recommends software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-get install -y python3.10 python3.10-dev python3.10-distutils
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
ln -s /usr/bin/python3.10 /usr/local/bin/python
```
