#!/bin/bash

rm -rf "$LOOPS_PATH/data/filtered/"
rm -rf "$LOOPS_PATH/data/unfiltered/"
rm -rf "$LOOPS_PATH/data/removed/"
python "$LOOPS_PATH/scripts/structure_filter_dataset.py"

bash "$LOOPS_PATH/scripts/find_duplicate_data.sh" > "$LOOPS_PATH/data/dupes_out.txt" 2> "$LOOPS_PATH/data/dupes_err.txt"

rm -rf "$LOOPS_PATH/data/processed/"
rm -rf "$LOOPS_PATH/data/processed_unfiltered/"
python "$LOOPS_PATH/scripts/process_loops_xmls.py" > /dev/null 2> /dev/null
python "$LOOPS_PATH/scripts/process_loops_xmls.py" "$LOOPS_PATH/data/unfiltered/" "$LOOPS_PATH/data/processed_unfiltered/" > "$LOOPS_PATH/data/process_out.txt" 2> "$LOOPS_PATH/data/process_err.txt"

rm -rf "$LOOPS_PATH/data/processed_512/"
bash "$LOOPS_PATH/scripts/resize_dir_imgs.sh"

python "$LOOPS_PATH/scripts/get_loops_csv.py"
python "$LOOPS_PATH/scripts/get_loops_csv.py" "$LOOPS_PATH/data/processed/" "$LOOPS_PATH/data/processed/"
python "$LOOPS_PATH/scripts/get_loops_csv.py" "$LOOPS_PATH/data/processed_unfiltered/" "$LOOPS_PATH/data/processed_unfiltered/"

printf "##### OUTPUT FILES #####\n\n"

cat "$LOOPS_PATH/data/dupes_out.txt"
cat "$LOOPS_PATH/data/dupes_err.txt"
printf "\n########################\n"
cat "$LOOPS_PATH/data/process_out.txt"
cat "$LOOPS_PATH/data/process_err.txt" | egrep "^From \".+\":    No Regions From Labeler \"[a-zA-Z]+\"$" -v

printf "\n########################\n"
