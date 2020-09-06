#data preprocessing for train-test-split in loan data set

import gzip
import pandas as pd
import numpy as np
from glob_param import *
import os

def split_gz():
    train_path = [ds_files(DS_DIR)[0], ds_files(DS_DIR)[2]]
    test_path = [ds_files(DS_DIR)[1], ds_files(DS_DIR)[3]]
    
    train_content = []
    test_content = []
    for fpath in train_path:
        #check how many lines in that file        
        with gzip.open(fpath, 'rb') as f:
            #print(len(f))
            for line in f:
                train_content.append(line)
    for fpath in test_path:
        with gzip.open(fpath, 'rb') as f:
            for line in f:
                test_content.append(line)
                
    with gzip.open(os.path.join(DS_SPLIT, 'ds_loan_2_train.vw.gz'), 'w') as f1:
        for i in range(len(train_content)):
            f1.write(train_content[i])
            
    with gzip.open(os.path.join(DS_SPLIT, 'ds_loan_2_test.vw.gz'), 'w') as f2:
        for i in range(len(test_content)):
            f2.write(test_content[i])
            
    all_content = train_content + test_content
    with gzip.open(os.path.join(DS_DIR, 'ds_loan_2.vw.gz'), 'w') as f3:
        for i in range(len(all_content)):
            f3.write(all_content[i])

def split_csv():
    df1 = pd.read_csv(DS_DIR + 'ds_loan1_2.csv')    
    df2 = pd.read_csv(DS_DIR + 'ds_loan2_2.csv')  
    df3 = pd.read_csv(DS_DIR + 'ds_loan3_2.csv')  
    df4 = pd.read_csv(DS_DIR + 'ds_loan4_2.csv')
    df_train = pd.concat([df1, df3])
    df_test = pd.concat([df2, df4])
    df_all = pd.concat([df_train, df_test])
    df_train.to_csv(DS_SPLIT + 'ds_loan_2_train.csv', index = None)
    df_test.to_csv(DS_SPLIT + 'ds_loan_2_test.csv', index = None)
    df_all.to_csv(DS_DIR + 'ds_loan_2.csv', index = None)

if __name__ == '__main__':
#    split_csv()
    split_gz()
