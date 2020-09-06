#reference to make_full.py 
#system param
import re
import csv
import numpy as np
import pandas as pd
def output_vw_csv(file_name, curr_qid = None, curr_num_docs = 0, curr_line = ''):
    
    pre_num_docs = max_docs
    for line in file_name:
        rel, qid, features = line.strip().split(maxsplit=2)
        if curr_qid is None:
            curr_qid = qid
        if curr_qid != qid:
            #write to the txt file
            ##indicate a new query id.
            file_handle.write(curr_line)
            
            file_handle.write('\n')
            curr_line = ''
            curr_num_docs = 0
            curr_qid = qid

        if curr_num_docs >= max_docs:
#            curr_row = [curr_num_docs, None]
#            writer.writerow(curr_row)
            continue
        elif curr_num_docs > 0:
            curr_line += ','
        
        curr_num_docs += 1
        #print(curr_num_docs, rel)
        curr_line += '{}:{} | '.format(curr_num_docs, 1. - 0.25 * float(rel)) + features
        #match for csv
        temp_re = features.split(' ')
        feature_name = [int(temp_re[i].split(':')[0]) for i in range(len(temp_re))]
        feature_full_value = np.zeros(feature_number)
        feature_value = [float(temp_re[i].split(':')[1]) for i in range(len(temp_re))]
        for i in range(len(feature_name)):
            if feature_name[i] <= feature_number:
                feature_full_value[feature_name[i] - 1] = feature_value[i] 
        curr_row = [curr_num_docs, 1 - 0.25*float(rel)] + list(feature_full_value)
        if curr_num_docs == 1 and pre_num_docs != max_docs:
            for i in range(pre_num_docs + 1, max_docs + 1):
                writer.writerow([i])
        writer.writerow(curr_row)
        pre_num_docs = curr_num_docs


    #write the remaining contents in the txt
    if curr_num_docs > 0:
        file_handle.write(curr_line)
        file_handle.write('\n')

##easy for our algorithm to process
#def csv_process(csvpath):
#    df = pd.read_csv(csvpath)
#    curr_num_docs = list(df['curr_num_docs'])
#    
#    for i in range(len(curr_num_docs) - 1):
#        if curr_num_docs[i] <= 9:
#            if curr_num_docs[i + 1] != curr_num_docs[i] + 1:
#                print(i, curr_num_docs[i])
#    #change the position
#    
#    df.to_csv(csvpath, index = None)


#MS = 10, yahoo = 6

#MS = 136, yahoo = 415

VW_DS_DIR = '../datasets/'
rank_type = 'yahoo'
if __name__ == '__main__':

    if rank_type == 'MS':
        max_docs = 10
        feature_number = 136
        folder = 1
        file_path = 'MSLR-WEB10K/Fold' + str(folder) + '/'
        file_path1 = file_path + 'test'
        file_path2 = file_path + 'vali'

    elif rank_type == 'yahoo':
        max_docs = 6
        folder = 2
        file_path = 'yahoo_dataset/ltrc_yahoo/set'
        file_path1 = file_path + str(folder) + '.test'
        file_path2 = file_path + str(folder) + '.valid'
        feature_number = 699


    curr_qid = None
    curr_num_docs = 0
    curr_line = ''

    #read in the csv FILE
    csvFile = open(VW_DS_DIR + file_path + str(folder) + "all.csv",'w',newline = "")
    writer = csv.writer(csvFile)
    columns = ['curr_num_docs', 'relevance'] + list(range(1, feature_number + 1))
    writer.writerow(columns)
    file_handle = open(VW_DS_DIR + file_path + str(folder) + ".txt",'w')

    file = open(VW_DS_DIR + file_path1 + '.txt')
    output_vw_csv(file)
    file = open(VW_DS_DIR + file_path2 + '.txt')
    output_vw_csv(file)
    
    file_handle.close()
    csvFile.close()
#csv_process(VW_DS_DIR + MSLR_file_path + "Fold" + str(MSLR_folder) + "1.csv")