import argparse
import configparser
import glob
import os

import numpy as np
from tqdm import tqdm


class Converter:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config_path = os.path.join(os.getcwd(), 'config.ini')
        self.config.read(self.config_path)
        self.npz_files = glob.glob(self.config['NPZ']['DATA_PATH'][1:-1] + '/**.npz')
        os.makedirs('./csvs', exist_ok=True)

    def convert(self, max_num):
        train_csv = open('./csvs/train_old.csv', 'w')
        test_csv = open('./csvs/test.csv', 'w')
        train_written = 0
        test_written = 0

        for file in tqdm(self.npz_files):

            subset, counter = (train_csv, train_written) if 'train' in file else (test_csv, test_written)
            if counter >= max_num:
                continue
            data = np.load(file)
            keys = list(data.keys())
            assert len(keys) == 1
            key = keys[0]
            data_to_write = data[key][:max_num]
            for i in tqdm(range(data_to_write.shape[0])):
                if counter >= max_num:
                    break
                subset.write(f'{data_to_write[i][0]},{data_to_write[i][1]}\n')
                counter += 1
            train_written = counter if 'train' in file else train_written
            test_written = counter if 'train' in file else test_written
        train_csv.close()
        test_csv.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_num', type=int, default=10000)
    args = parser.parse_args()
    c = Converter()
    c.convert(args.max_num)
