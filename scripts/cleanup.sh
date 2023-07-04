#!/bin/bash

export in_dir=${1:-"$LOOPS_PATH"}

find "$in_dir" -name "__pycache__" -type d | xargs -I{} rm -r "{}"
find "$in_dir" -name ".ipynb_checkpoints" -type d | xargs -I{} rm -r "{}"
