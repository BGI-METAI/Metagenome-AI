#!/bin/bash

# Input and base output path
input_file="/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/01_origin/non_anno/full_rep_seq.true_orf.fasta"
output_dir="/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/chunks/full_rep_seq.true_orf/chunk100w"
num=1000000  # Number of records per file

# Create output directory if it doesn't exist
mkdir -p $output_dir

# Verify output directory is set correctly
echo "Output directory is set to: $output_dir"
if [ ! -d "$output_dir" ]; then
    echo "Error: Output directory does not exist."
    exit 1
fi

# Process the file
awk -v RS=">" -v num="$num" -v output_dir="$output_dir" -F '[;#\n]' 'NR>1 {
    # Trim leading and trailing spaces from each field
    for (i=1; i<=NF; i++) {
        gsub(/^[\t ]+|[\t ]+$/, "", $i);
    }

    # Build the output content for each record
    output = sprintf("%s,%s,%s,%s,", $1, $2, $3, $4);
    for (i=5; i<11; i++) {
        split($i, arr, "=");
        output = output sprintf("%s,", arr[2]);
    }
    for (i=11; i<NF; i++) {
        if ($i ~ /^[A-Z]+/) {
            output = output sprintf("%s", $i);
        }
    }

    # Decide which file to write to based on the count of records
    if (NR % num == 2 || NR == 2) {
        if (out_file) {
            close(out_file);
        }
        file_num = int((NR-1)/num) + 1;
        out_file = sprintf("%s/chunk%02d.txt", output_dir, file_num);
    }
    print output > out_file;
}' $input_file