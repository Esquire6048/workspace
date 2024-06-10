# log_for_LLM

## GPT


## Bert

text classification, text summarization, and more.

https://medium.com/@karkar.nizar/fine-tuning-bert-for-text-classification-with-lora-f12af7fa95e4

## ALpaca


## Llama


## LORA

fine-tuning a large language model (LLM) from scratch requires significantly more parameters than when fine-tuning from a pre-trained model.

LORA introduces a method to learn a lower-dimensional, task-specific representation of the layer’s weights

Consider a fully connected layer with ‘m’ input units and ’n’ output units. The weight matrix for this layer has dimensions ‘m x n’. When we provide an input ‘x’, the output of this layer is calculated using the formula Y = W X.

During fine-tuning with LORA, we keep ‘W’ fixed and introduce two matrices, ‘A’ and ‘B’, into the equation. The new equation becomes Y = W X + A*B X. 

![LORA Matrix](https://miro.medium.com/v2/resize:fit:640/format:webp/1*d1ckUy_f3nfdTP_J0xzs-g.png "LORA Matrix")


## Docker

ssh z-xu@192.168.147.99

docker images

docker run -it -v $(pwd)/pytorch_simplest_tutorial/:/home/tutorial/ --name "docker_practice" --gpus all  --rm sample_image_xu bash

docker run -it -v \$(pwd)/pytorch_simplest_tutorial/:/home/tutorial/ --name "docker_practice" --gpus all -v \$(pwd)/models:/root/models/ --rm sample_image_xu bash

