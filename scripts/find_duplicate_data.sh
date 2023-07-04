#!/bin/bash

export in_dir=${1:-"$LOOPS_PATH/data/unfiltered/"}

find "$in_dir" -type f | egrep -i "\.(png|tif|tiff|jpg|jpeg)$" | rev | cut -d'/' -f1 | cut -d'.' -f2- | rev | sort | uniq -D

findimagedupes --rescan --threshold=85% --recurse "$in_dir" 2>&1 | egrep "^Warning: not fingerprinting unknown-type file: .+\.(xml|xlsx|pptx)$" -v | sort

echo ""
tmpfile=$(mktemp /tmp/rdfind.XXXXXX)
rdfind -ignoreempty false -minsize 0 -followsymlinks true -checksum md5 -deterministic true -makeresultsfile true -outputname "$tmpfile" -dryrun true "$in_dir"
cat "$tmpfile"
rm "$tmpfile"
