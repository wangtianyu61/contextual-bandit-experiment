# test how the algorithm works for train_test_split
import numpy as np
import pandas as pd
#cannot import such KEY
#from config import OML_API_KEY
import gzip
import os
import random
import csv
#from vowpalwabbit import pyvw
from run_vw_job import process, params, param_grid, expand_cover
from split_gz_csv import *
from glob_param import *
split_percent = 0.5

def ds_files(filepath):
    import glob
    return sorted(glob.glob(os.path.join(filepath, '*.vw.gz')))

def sample_custom_pmf(pmf):
    total = sum(pmf)
    scale = 1 / total
    pmf = [x * scale for x in pmf]
    draw = random.random()
    sum_prob = 0.0
    for index, prob in enumerate(pmf):
        sum_prob += prob
        if(sum_prob > draw):
            return index, prob


                
#classfication for each algorithm
all_params = param_grid()
epsilon_greedy_class = []
greedy_class = []
cover = []
cover_nu = []
bag = []
bag_greedy = []
supervised = []

for item in all_params:
    if item['alg'][0] == 'epsilon':
        if item['alg'][1] == 0:
            greedy_class.append(item)
        else:
            epsilon_greedy_class.append(item)
    elif item['alg'][0] == 'supervised':
        supervised.append(item)
    elif item['alg'][0] == 'bag':
        if len(item['alg']) == 4:
            bag.append(item)
        else:
            bag_greedy.append(item)
    elif item['alg'][0] == 'cover':
        if len(item['alg']) == 6:
            cover_nu.append(item)
        elif len(item['alg']) == 2:
            cover.append(item)
        else:
            if item['alg'][2] == 'psi':
                cover.append(item)
            else:
                cover_nu.append(item)
        
def best_params_selection(did, alg_class):    
    #select params from different algorithms
    pv_loss = np.zeros(len(alg_class))
    for i in range(len(alg_class)):
        pv_loss[i] = process(ds_files(DS_SPLIT)[2*did + 1], alg_class[i], None)[0]
    #use them for training data.                
    #select the best hyperparams for the test dataset.
    pv_loss_min = np.where(pv_loss == np.min(pv_loss))[0]
    if len(pv_loss_min) == 1:
        index = pv_loss_min[0]
    else:
        #randomtie
        index, prob = sample_custom_pmf(list(np.ones(len(pv_loss_min))))
        #the true index in the original model
        index = pv_loss_min[index]
    return index, pv_loss[index]

def validation_test(did, alg):
    pv_loss_all = process(ds_files(DS_DIR)[did], alg, None)[0]
    return pv_loss_all

def train_test_process():
    csv_test = open(DS_RESULT + 'test' + str(split_percent) +'.csv', 'a', newline = '')
    csv_writer = csv.writer(csv_test)
    did_num = ds_files(DS_DIR)
    did_list = [os.path.basename(a).split('.')[0].split('_')[1] for a in ds_files(DS_DIR)]
    param_head = ['alg', 'alg_param_number'] + did_list
    csv_writer.writerow(param_head)


#    pv_loss_test = np.zeros(len(did_num))
#    for i in range(len(did_num)):
#        index, pv_loss_train = best_params_selection(i, supervised)
#        pv_loss_test[i] = (validation_test(i, supervised[index]) - split_percent*pv_loss_train)/(1 - split_percent)
#    csv_writer.writerow(['supervised', len(supervised)] + list(pv_loss_test))

    pv_loss_test = np.zeros(len(did_num))
    for i in range(len(did_num)):
        index, pv_loss_train = best_params_selection(i, epsilon_greedy_class)
        pv_loss_test[i] = (validation_test(i, epsilon_greedy_class[index]) - split_percent*pv_loss_train)/ (1 - split_percent)
    csv_writer.writerow(['epsilon-greedy', len(epsilon_greedy_class)] + list(pv_loss_test))

    pv_loss_test = np.zeros(len(did_num))
    for i in range(len(did_num)):
        index, pv_loss_train = best_params_selection(i, greedy_class)
        pv_loss_test[i] = (validation_test(i, greedy_class[index]) - split_percent*pv_loss_train)/(1 - split_percent)
    csv_writer.writerow(['greedy', len(greedy_class)] + list(pv_loss_test))

    pv_loss_test = np.zeros(len(did_num))
    for i in range(len(did_num)):
        index, pv_loss_train = best_params_selection(i, cover)
        pv_loss_test[i] = (validation_test(i, cover[index]) - split_percent*pv_loss_train)/(1 - split_percent)
    csv_writer.writerow(['cover', len(cover)] + list(pv_loss_test))

    pv_loss_test = np.zeros(len(did_num))
    for i in range(len(did_num)):
        index, pv_loss_train = best_params_selection(i, cover_nu)
        pv_loss_test[i] = (validation_test(i, cover_nu[index]) - split_percent*pv_loss_train)/(1 - split_percent)
    csv_writer.writerow(['cover-nu', len(cover_nu)] + list(pv_loss_test))

    pv_loss_test = np.zeros(len(did_num))
    for i in range(len(did_num)):
        index, pv_loss_train = best_params_selection(i, bag)
        pv_loss_test[i] = (validation_test(i, bag[index]) - split_percent*pv_loss_train)/(1 - split_percent)
    csv_writer.writerow(['bag', len(bag)] + list(pv_loss_test))

    pv_loss_test = np.zeros(len(did_num))
    for i in range(len(did_num)):
        index, pv_loss_train = best_params_selection(i, bag_greedy)
        pv_loss_test[i] = (validation_test(i, bag_greedy[index]) - split_percent*pv_loss_train)/(1 - split_percent)
    csv_writer.writerow(['bag-greedy', len(bag_greedy)] + list(pv_loss_test))

    csv_test.close()

if __name__ == '__main__':
    i = 0
    #train_test_split_gz(split_percent)
    #train_test_split_csv(split_percent)
    train_test_process()
