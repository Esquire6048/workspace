# Docker

## Common cmd

显示所有镜像
`docker images`

显示所有容器
`docker ps -a`

运行容器
`docker run -it --name="docker_practice" -v=$(pwd)/pytorch_simplest_tutorial/:/home/tutorial/ -v=$(pwd)/models:/root/models/ --gpus all --rm sample_image_xu bash`

-it 交互式

-p=host_port:container_port 映射端口

-v=host_directory:container_directory 挂载数据

--rm 退出容器后删除容器