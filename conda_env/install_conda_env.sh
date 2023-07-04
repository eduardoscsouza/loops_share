#!/bin/bash

env_name="${1:-"loops"}"
project_path="${2:-"$HOME/Documents/code/loops/"}"
project_var_name="${3:-"LOOPS_PATH"}"

#Hook Shell so that the Environment can be Activated
eval "$(command conda 'shell.bash' 'hook')"
set -Eeo pipefail

#Create and Activate Environment
conda create --yes --name "$env_name"
conda activate "$env_name"
conda env config vars set "$project_var_name"="$project_path"

#Install Package YMLs
conda env update --file base_conda_packages.yml
conda env update --file tf_2_10_conda_packages.yml
conda env update --file pytorch_1_13_conda_packages.yml
conda env update --file yolov5_conda_packages.yml
conda env update --file loops_conda_packages.yml

#Fix to matplotlib's "qt.qpa.plugin: Could not load the Qt platform plugin "xcb""
pip uninstall --yes opencv-python
pip install opencv-python-headless
