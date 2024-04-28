import os
import random
import unittest

random.seed(42)


class TestDataProcess(unittest.TestCase):
    def test_split_train_test(self):
        # split train and test data set
        data_path = "/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/02_pkls"
        save_path = "/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/"
        train_rate = 0.7
        write_num = 100

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
