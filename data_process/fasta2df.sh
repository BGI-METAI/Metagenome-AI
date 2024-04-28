input_file="/home/share/huadjyin/home/yinpeng/ljl/data/METAdataset/data/non_anno/oral_saliva_nonanno_seq.fasta"
output_file="/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/train/chunk10w/chunk01.txt"
num=100000

awk -v RS=">" -F '[;#\n]' 'NR>1 {
    printf "%s %s %s %s ", $1, $2, $3, $4;
    for (i=5; i<11; i++) {
        split($i, arr, "=");
        printf "%s ", arr[2];
    }
    for (i=11; i<NF; i++) {
        if ($i ~ /^[A-Z]+/) {
            printf "%s", $i;
        }
    }
    printf "\n";
}' $input_file | head -n $num > $output_file

# split to small file
