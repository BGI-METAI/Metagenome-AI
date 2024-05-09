import logging
import argparse
import os
import os.path as osp
import pickle

from tqdm import tqdm


def register_parameters():
    parser = argparse.ArgumentParser(description='Protein Sequence Amino Acids Annotation Framework')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./',
        help='the path of fasta file by processed to df'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/02_pkls',
        help='the path of output'
    )

    return parser.parse_args()


def process_line(line, save_path):
    # protein_id, #1, #2, #3, #4, partial, start_type, rbs_motif, rbs_spacer, gc_count, sequence
    # protein_id, _, _, _, _, partial, _, _, _, gc_count, seq = line.strip().split(' ')
    values = []
    for value in line.strip().split(' '):
        if value.strip():
            values.append(value.strip())
    protein_id, seq, partial, gc_count = values[0], values[-1], values[5], values[9]
    # remove "*" in ORF
    seq = seq[:-1]
    logging.debug(f"protein_id:{protein_id}, seq:{seq}, partial:{partial}, gc_count:{gc_count}")
    with open(osp.join(save_path, f'{protein_id}.pkl'), 'wb') as file:
        pickle.dump({'seq': seq, "partial": partial, "gc_count": gc_count}, file)


def process_file(data_path: str, file: str, save_path: str):
    with open(osp.join(data_path, file), 'r') as fp:
        lines = fp.readlines()
        # partial_func = partial(process_line, save_path=save_path)
        # with Pool() as pool:
        #     pool.map(partial_func, lines)
        for line in lines:
            process_line(line, save_path)


def main():
    args = register_parameters()
    logging.debug(f"fasta2pkl| {args}")
    assert os.path.exists(args.data_dir), f"{args.fasta_df} not exit"
    assert os.path.exists(args.output_dir), f"{args.output_dir} not exit"

    for file in tqdm(os.listdir(args.data_dir)):
        if not file.endswith('.txt'):
            continue

        save_folder_name = file.split('.')[0]
        save_folder_path = osp.join(args.output_dir, save_folder_name)
        print(f"process file: {os.path.join(args.data_dir, file)}")
        os.makedirs(save_folder_path, exist_ok=True)

        process_file(args.data_dir, file, save_folder_path)


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    main()
