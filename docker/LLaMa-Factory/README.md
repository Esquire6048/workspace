docker build -f $(pwd)Dockerfile --build-arg INSTALL_BNB=false --build-arg INSTALL_VLLM=false --build-arg INSTALL_DEEPSPEED=false --build-arg PIP_INDEX=https://pypi.org/simple -t llamafactory:latest .

docker run -it --gpus=all -v $(pwd)/hf_cache:/root/.cache/huggingface/ -v $(pwd)/data:/app/data  -v $(pwd)/output:/app/output -p 7860:7860 -p 8000:8000 --shm-size 16G --name llamafactory llamafactory:latest

llamafactory-cli webui
