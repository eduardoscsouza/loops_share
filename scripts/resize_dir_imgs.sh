#!/bin/bash

export in_dir=${1:-"$LOOPS_PATH/data/processed/"}
export out_dir=${2:-"$LOOPS_PATH/data/processed_512/"}
export out_size=${3:-"512x512"}

cp -p -r "$in_dir" "$out_dir"
find "$in_dir" -type f | grep -oP "^$in_dir\K.*" | xargs -I{} printf 'no_format=$(printf "{}" | rev | cut -d. -f2- | rev); convert -filter Cubic -resize "$out_size!" "$in_dir/{}" "$out_dir/$no_format.png" && if [ "$out_dir/{}" != "$out_dir/$no_format.png" ]; then rm "$out_dir/{}"; fi;\n' | parallel
