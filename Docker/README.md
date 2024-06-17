# Docker

## Common cmd

显示所有镜像

`docker images`

显示所有容器

`docker ps -a`

新建镜像

`docker build -t docker_name .`

运行容器

`docker run -it --name="container_name" -v=$(pwd)/:/root/ --gpus all --rm image_name bash`

-it 交互式

-p=host_port:container_port 映射端口

-v=host_directory:container_directory 挂载数据

--rm 退出容器后删除容器
