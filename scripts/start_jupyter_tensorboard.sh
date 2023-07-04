#!/bin/bash

env_name="${1:-"loops"}"
jupyter_root_dir="${2:-"$LOOPS_PATH"}"
tensorboard_logdir="${3:-"$LOOPS_PATH/experiments/tensorboard_logs/"}"

#Hook Shell so that the Environment can be Activated
eval "$(command conda 'shell.bash' 'hook')"
conda activate "$env_name"

#Trap child processes so they are killed on exit
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

cd "$jupyter_root_dir"
jupyter lab &
tensorboard --logdir "$tensorboard_logdir" &
wait
