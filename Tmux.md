# Tmux

鼠标模式
touch ~/.tmux.conf
set -g mode-mouse on
set -g mouse-select-pane on

新建会话

tmux new -s [name]

列出会话

tmux ls

恢复上一次对话

tmux

恢复指定名字会话

tmux a -t [name]

退出tmux进程
⌃b d

翻页模式
⌃b [ 
q
