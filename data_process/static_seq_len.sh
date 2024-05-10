#!/bin/bash

# Input and base output path
directory="/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/chunks/full_rep_seq.true_orf/chunk100w"
output_file="/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/chunks/full_rep_seq.true_orf/static_len_100w.csv"

# Check if the directory exists
if [ ! -d "$directory" ]; then
    echo "Error: Directory does not exist: $directory"
    exit 1
fi

# Prepare the output file by writing a header (optional)
echo "name,seq" > $output_file

# Find all .txt files and process them
find "$directory" -name "*.txt" -print0 | while IFS= read -r -d $'\0' file
do
    echo "Processing file: $file"
    awk -F, '{
        name = $1; # Extract name
        seq = $NF; # Extract sequence, where $NF is the last field
        partial = $6; # Whether there is no partial sequence
        gc_count = $10; # gc count
        len= length($seq) - 1
        print name "," len "," partial "," gc_count
    }' "$file" >> $output_file
done

echo "Extraction complete. Results are saved in $output_file"
