import os
import pickle
import random
import unittest

import numpy as np

random.seed(42)


class TestDataProcess(unittest.TestCase):
    def test_split_train_test(self):
        # split train and test data set
        data_path = "/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/02_pkls/full_rep_seq.true_orf"
        save_path = "/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/full_rep_seq.true_orf/chunk100w"
        train_rate = 0.8
        write_num = 100000
        total_samples = 1000000

        assert os.path.exists(data_path), f"{data_path} is not exits"

        train_datas = []
        test_datas = []
        i = 0
        for f_path, dirs, files in os.walk(data_path):
            if i > total_samples:
                break

            for file in files:
                if file.endswith("pkl") is False:
                    continue
                file_path = os.path.join(f_path, file)
                if random.random() < train_rate:
                    train_datas.append(file_path)
                    i += 1  # Note: make sure total sample with train dataset
                else:
                    test_datas.append(file_path)

                if len(train_datas) > write_num:
                    with open(os.path.join(save_path, "train.txt"), "a") as f:
                        f.write("\n".join(train_datas) + "\n")
                        train_datas = []

                if len(test_datas) > write_num:
                    with open(os.path.join(save_path, "test.txt"), "a") as f:
                        f.write("\n".join(test_datas) + "\n")
                        test_datas = []

        if train_datas:
            with open(os.path.join(save_path, "train.txt"), "a") as f:
                f.write("\n".join(train_datas) + "\n")
        if test_datas:
            with open(os.path.join(save_path, "test.txt"), "a") as f:
                f.write("\n".join(test_datas) + "\n")

    def test_check_pkl(self):
        """check pkl file"""
        import pandas as pd
        train_path = "/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/train.txt"
        save_dir = "/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/"

        names, lens = [], []
        with open(train_path, 'r') as f:
            samples = f.readlines()
            for sample in samples:
                sample = sample.strip()
                try:
                    with open(sample, 'rb') as fp:
                        data = pickle.load(fp)
                        names.append(os.path.basename(sample))
                        lens.append(len(data["seq"]))
                except Exception as e:
                    print(f"{sample} was error: {e}")
        df = pd.DataFrame({"name": names, "lens": lens})
        df.to_csv(os.path.join(save_dir, "test_analysis.csv"))

    def test_static_dis_with_len(self):
        # 获取含有不同蛋白的pfam的对应蛋白，用于做embedding可视化
        """
        pfam, protein_num, protein_name, protein_seq

        :return:
        """
        pfam_nums = [i for i in range(100, 20000, 2000)]
        cal_values = [-1 for _ in range(10)]
        cal_name = ["" for _ in range(10)]
        protein_names_all =[[] for _ in range(10)]
        protein_seq_all = [[] for _ in range(10)]
        thread = 500
        pfam_statistic_path = "/home/share/huadjyin/home/yinpeng/zkx/data/interpro/interpro_result/pFAM_info_out.txt"
        pfam_protein_family_path = "/home/share/huadjyin/home/yinpeng/zkx/data/interpro/interpro_result/pFAM_info_out.txt"
        pfam_protein_seq_path = "/home/share/huadjyin/home/yinpeng/zkx/data/interpro/interpro_result/pFAM_info_out.txt"

        print(len(pfam_nums))

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx], idx

        # 获取对应的pfam label
        with open(pfam_statistic_path, "r") as f:
            for pf_value in f.readlines():
                name, value = pf_value.split(" ")
                name = name.strip()
                value = int(value.strip())
                near_value, idx = find_nearest(pfam_nums, value)
                offset = abs(near_value - value)
                if offset < thread and abs(cal_values[idx] - value) > offset:
                    cal_values[idx] = value
                    cal_name[idx] = name

                if -1 not in cal_values:
                    break

        # 获取对应的protein name
        with open(pfam_protein_family_path, "r") as f:
            for pf_value in f.readlines():
                values = [name.strip() for name in pf_value.split(" ")]
                family_name = values[0]
                protein_names = values[1:]
                if family_name in cal_name:
                    protein_names_all.append(protein_names)

        # 获取对应的seq
        with open(pfam_protein_seq_path, "r") as f:
            for pf_value in f.readlines():
                values = [name.strip() for name in pf_value.split(" ")]
                protein_name = values[0]
                protein_seq = values[2]
                for i in range(len(protein_names_all)):
                    if protein_name in protein_names_all[i]:
                        protein_seq_all[i].append(protein_seq)

        for name, value in zip(cal_name, cal_values):
            print(f"name: {name} -> nums: {value}")
