# Common commands

<details>
  <summary>Nvidia</summary>

```bash
nvidia-smi
```

```bash
top -p $(nvidia-smi | grep -oP '\d+(?=\s+C)' | tr '\n' ',' | sed 's/,$//')
```

```bash
export CUDA_VISIBLE_DEVICES=1
```

</details>

<details>
  <summary>Tmux</summary>

新建会话 `tmux new -s [name]`

列出会话 `tmux ls`

恢复上一次对话 `tmux`

恢复指定名字会话 `tmux a -t [name]`

退出tmux进程 `⌃b d`

翻页模式 `⌃b [ ` 退出翻页模式 `q`

</details>

<details>
  <summary>Docker</summary>

显示所有镜像  `docker images`

显示所有容器 `docker ps -a`

新建镜像 `docker build -t docker_name .`

运行容器 `docker run -it --name="container_name" -v=$(pwd)/:/root/ --gpus all --rm image_name bash`

-it 交互式

-p=host_port:container_port 映射端口

-v=host_directory:container_directory 挂载数据

--rm 退出容器后删除容器

恢复已停止容器 `docker start -a [ID]`

删除容器 `docker rm [ID]`

删除镜像 `docker rmi [ID]`

</details>
