#!/bin/bash

# Ensure the input file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

# Ensure the output directory is provided
if [ -z "$2" ]; then
  echo "Usage: $0 <input_file> <output_dir>"
  exit 1
fi

input_file="$1"
output_dir="$2"
total_lines=$(wc -l < "$input_file")
lines_per_file=$((total_lines / 10))
extra_lines=$((total_lines % 10))

split -l $lines_per_file "$input_file" part_

# Rename the files to 0.txt, 1.txt, ..., 9.txt
i=0
for file in part_*; do
  mv "$file" "${i}.txt"
  i=$((i + 1))
done

cd "$output_dir"
# Distribute extra lines
if [ $extra_lines -ne 0 ]; then
  awk 'NR > '"$((total_lines - extra_lines))"' { print > (FILENAME)NR%10".txt" }' FILENAME="${input_file%.*}" "$input_file"
fi