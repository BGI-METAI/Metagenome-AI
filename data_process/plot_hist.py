import argparse
import os.path

import numpy as np
import matplotlib.pyplot as plt

def register_parameters():
    parser = argparse.ArgumentParser(description='Statistics the sequence length distribution of the specified column')
    parser.add_argument(
        '--file_path',
        type=str,
        default='/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/chunks/full_rep_seq.true_orf/static_len_100w.csv',
        help='the path of input dataset'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/chunks/full_rep_seq.true_orf/statistic_analysis/ORF_fragment_length_all_110w.png',
        help='save path of hist'
    )
    parser.add_argument(
        '--col_name',
        type=str,
        default=None,
        help='column name'
    )
    parser.add_argument(
        '--col_index',
        type=int,
        default=0,
        help='column index'
    )
    parser.add_argument(
        '--max_len',
        type=int,
        default=np.inf,
        help='max sequence len for statistic analysis'
    )
    parser.add_argument(
        '--separator',
        type=str,
        default=",",
        help='separator'
    )
    return parser.parse_args()


def get_seq_distribution(file_path: str, col_name: str = None, col_idx: int = 0, max_len: int = np.inf,
                         separator: str = ","):
    """
    get seq distribution
    :param file_path: 文件路径
    :param max_len: 最大统计长度
    :param col_name: 统计那一列名
    :param col_idx: 统计第几列
    :param split: 分隔符
    :return:
    """
    assert os.path.exists(file_path), f"`{file_path}` is not exits"
    normal_length_list = []
    unnormal_length_list = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        first_line = lines[0].split(separator)

        if col_name in first_line:
            lines = lines[1:]
            col_idx = first_line.index(col_name)
            print(f"col name: {first_line}\n get col index: {col_idx}")
        else:
            assert col_idx <= len(first_line), f"{col_idx} out of max column ({len(first_line)} "
            print(f"statistic value: {first_line[col_idx]}")

        for line in lines:
            value = int(line.split(separator)[col_idx].strip())
            if value > max_len:
                unnormal_length_list.append(value)
            else:
                normal_length_list.append(value)
    print("out of length list nums: ", len(unnormal_length_list))
    print("normal length list nums: ", len(normal_length_list))
    return normal_length_list, unnormal_length_list


def draw_histogram(normal_length_list: list, save_path: str):
    """
    draw histogram
    :param normal_length_list: values of list
    :param save_path: save path
    :return: None
    """
    assert os.path.exists(os.path.dirname(save_path)), f"`{save_path}` dir is not exits."

    normal_length_list = np.array(normal_length_list)
    percentile_95 = np.percentile(normal_length_list, 95)
    percentile_5 = np.percentile(normal_length_list, 5)
    min_length = min(normal_length_list)
    max_length = max(normal_length_list)
    print('最大值为 %d，最小值为 %d' % (max_length, min_length))
    print('95分位数为 %d, 5分位数为 %d' % (percentile_95, percentile_5))
    hist1, bin_edges1 = np.histogram(normal_length_list, bins=100, range=(min_length, max_length))
    fig1, ax1 = plt.subplots()
    ax1.hist(bin_edges1[:-1], bin_edges1, weights=hist1, facecolor='blue',
             alpha=0.7, edgecolor='k')
    # 修改科学计数法为整数，不为1e6
    # ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax1.axvline(percentile_95, color='red', linestyle='--', label='95th percentile')
    ax1.axvline(percentile_5, color='green', linestyle='--', label='5th percentile')
    ax1.legend()
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Fragment Counts')
    plt.savefig(save_path)
    print(f"save hist path: `{save_path}`")


def main():
    args = register_parameters()
    logging.debug(args)

    normal_length_list, unnormal_length_list = get_seq_distribution(
        file_path=args.file_path,
        col_name=args.col_name,
        col_idx=args.col_index,
        max_len=args.max_len,
        separator=args.separator

    )
    draw_histogram(normal_length_list, args.save_path)


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
