import os
import pickle
import random
import unittest

random.seed(42)


class TestDataProcess(unittest.TestCase):
    def test_split_train_test(self):
        # split train and test data set
        data_path = "/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/02_pkls"
        save_path = "/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets"
        train_rate = 0.7
        write_num = 1000

        assert os.path.exists(data_path), f"{data_path} is not exits"

        train_datas = []
        test_datas = []
        for f_path, dirs, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(f_path, file)
                if random.random() < train_rate:
                    train_datas.append(file_path)
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
