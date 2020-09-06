"""
Helper script for mslr/yahoo learning-to-rank datasets. To be used as follows (for 10 different shuffles):

    ### MSLR
    cat train.txt vali.txt test.txt | python make_full.py > train_full.txt
    for i in {1..10}; do shuf vw_full.txt > vw_full$i.txt; done
    for i in {1..10}; do cat vw_full$i.txt | python full_to_ldf.py | gzip > vw_full$i.vw.gz; done
    for i in {1..10}; do cp vw_full$i.vw.gz mslr_shufs/ds_mslr${i}_10.vw.gz; done

    ### Yahoo
    cat set*.txt | python make_full.py --max_docs 6 > vw_full.txt
    for i in {1..10}; do shuf vw_full.txt > vw_full$i.txt; done
    for i in {1..10}; do cat vw_full$i.txt | python full_to_ldf.py | gzip > vw_full$i.vw.gz; done
    for i in {1..10}; do cp vw_full$i.vw.gz yahoo_shufs/ds_yahoo${i}_6.vw.gz; done

note: for simulating bandit feedback, use the --cbify_ldf option in VW
"""

import gzip
import os

VW_DS_DIR = '../datasets/yahoo_dataset/'

MSLR_folder = 5
MSLR_file_name = 'Fold' + str(MSLR_folder) + '/Fold' + str(MSLR_folder)

yahoo_folder = 2
yahoo_file_name = 'set'
file = open(VW_DS_DIR + yahoo_file_name + str(yahoo_folder) + ".txt")
with gzip.open(os.path.join(VW_DS_DIR, 'yahoo' + str(yahoo_folder) + '_6.vw.gz'), 'w') as f:
    count = 0
    for line in file:
         for lab_line in line.strip().split(','):
             if count > 1 and lab_line[0:2] == '1:':
                 f.write('\n'.encode())
             f.write(lab_line.encode())
             f.write("\n".encode())
             count = count + 1
                 
             