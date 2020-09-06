import os, gzip
import pandas as pd
from glob_param import *


def ds_files(filepath):
    import glob
    return sorted(glob.glob(os.path.join(filepath, '*.vw.gz')))


def ds_files_csv(filepath):
    import glob
    return sorted(glob.glob(os.path.join(filepath, '*.csv')))

#realize the function to split the train and test dataset
def train_test_split_gz(split_percent):
    for fpath in ds_files(DS_DIR):
        #check how many lines in that file
        count = 0
        with gzip.open(fpath, 'rb') as f:
            f_content = []
            #print(len(f))
            for line in f:
                f_content.append(line)
                if line.decode()[0] == '1' and line.decode()[1] == ':':                
                    count = count + 1
            split_point = count*split_percent
            print(count, split_point)
        fpath = fpath.strip('.vw.gz')
        fname = fpath.strip(DS_DIR + '\\')
        print(fname)
        break_point = 0
        count2 = 0
        with gzip.open(os.path.join(DS_SPLIT, 'ds' + fname + '_train.vw.gz'), 'w') as f1:
            for i in range(len(f_content)):
                if f_content[i].decode()[0] == '1' and f_content[i].decode()[1] == ':':
                    count2 = count2 + 1
                if count2 <= split_point:
                    f1.write(f_content[i])
                else:
                    break_point = i
                    break
        with gzip.open(os.path.join(DS_SPLIT, 'ds' + fname + '_test.vw.gz'), 'w') as f2:
            for i in range(break_point, len(f_content)):
                f2.write(f_content[i])

#realize the function to split the train and test dataset into csv format for FALCON to use
def train_test_split_csv(split_percent):
    doc_num = 6
    for csv_path in ds_files_csv(DS_DIR):
        df = pd.read_csv(csv_path)
        csv_path = csv_path.strip('.csv')
        csv_new_path = csv_path.strip(DS_DIR + '\\')
        print(csv_new_path)
        split_point = int(len(df)/doc_num)*split_percent
        df_train = df[0:int(split_point)*doc_num]
        df_train.to_csv(os.path.join(DS_SPLIT, 'ds' + csv_new_path + '_train.csv'))
        df_test = df[int(split_point)*doc_num:len(df)]
        df_test.to_csv(os.path.join(DS_SPLIT, 'ds' + csv_new_path + '_test.csv'))



