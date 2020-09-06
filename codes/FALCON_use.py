from FALCON import FALCON, FALCON_ldf, FALCON_price
from split_gz_csv import ds_files_csv
from split import sample_custom_pmf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from glob_param import *

split_percent = 0.4

    
#implement all the hyperparameters' combination of FALCON algorithm
def implement_all(comb, alg_type, min_did = 0, max_did = None, funclass = 'linear'):
    if max_did == None:
        csv_path_all = ds_files_csv(DS_DIR)[min_did: ]
    else:
        csv_path_all = ds_files_csv(DS_DIR)[min_did: max_did]
    loss_comb = np.zeros((len(comb), len(csv_path_all)))
    sum_loss_comb = np.zeros(len(comb))
    print(csv_path_all)

    for did in range(len(csv_path_all)):
        doc_num = int(csv_path_all[did].split('.')[2].split('_')[2])
        for i in range(len(comb)):
            if alg_type == 'cbify':
                sample_falcon = FALCON(csv_path_all[did], comb[i][0], comb[i][1], funclass)
            elif alg_type == 'cbify_ldf':
                sample_falcon = FALCON_ldf(csv_path_all[did], comb[i][0], comb[i][1], 6, funclass)
            elif alg_type == 'cbify_price':
                sample_falcon = FALCON_price(csv_path_all[did], comb[i][0], comb[i][1], doc_num, funclass)
            loss_comb[i][did] = sample_falcon.learn_schedule()
            print(loss_comb[i][did])
            sum_loss_comb[i] += loss_comb[i][did]
    #return sum_loss_comb, loss_comb
    index = np.where(sum_loss_comb == np.min(sum_loss_comb))[0][0]
    
    return index, loss_comb
        
        

    
#implement use the train_test_split_dataset with the best hyperparameters
def implement_train_test_split(comb, alg_type, doc_num = 6, min_did = 0, max_did = None):
    if max_did == None:
        csv_path_train = ds_files_csv(DS_SPLIT)[2*min_did: ]
        csv_path_all = ds_files_csv(DS_DIR)[min_did: ]
    else:
        csv_path_train = ds_files_csv(DS_SPLIT)[2*min_did: 2*max_did]
        csv_path_all = ds_files_csv(DS_DIR)[min_did : max_did]
    print(csv_path_all, csv_path_train)
    pv_loss_test = np.zeros(len(csv_path_all))
    for did in range(len(csv_path_all)):
        doc_num = str(csv_path_all[did].split('.')[2].split('_')[2])
        pv_loss = np.zeros(len(comb))
        for i in range(len(comb)):
            if alg_type == 'cbify':
                sample_falcon = FALCON(csv_path_train[2*did + 1], comb[i][0], comb[i][1])
            elif alg_type == 'cbify_ldf':
                sample_falcon = FALCON_ldf(csv_path_train[2*did + 1], comb[i][0], comb[i][1], doc_num, 'ridge')
            elif alg_type == 'cbify_price':
                sample_falcon = FALCON_price(csv_path_train[2*did + 1], comb[i][0], comb[i][1], doc_num, 'ridge')
            pv_loss[i] = sample_falcon.learn_schedule()
        print(pv_loss)
        pv_loss_min = np.where(pv_loss == np.min(pv_loss))[0]
        if len(pv_loss_min) == 1:
            index = pv_loss_min[0]
        else:
            #randomtie
            index, prob = sample_custom_pmf(list(np.ones(len(pv_loss_min))))
            index = pv_loss_min[index]
            
        #the true index in the original model
        if alg_type == 'cbify':
            falcon_true = FALCON(csv_path_all[did], comb[index][0], comb[index][1])
        elif alg_type == 'cbify_ldf':
            falcon_true = FALCON_ldf(csv_path_all[did], comb[index][0], comb[index][1], doc_num, 'ridge')
        pv_loss_all = falcon_true.learn_schedule()
        print(pv_loss[index], pv_loss_all)
        pv_loss_test[did] = (pv_loss_all -pv_loss[index]*split_percent)/(1 - split_percent)
    
    return pv_loss_test
    
#all the combinations of hyperparameters
def param():
    gamma = list(range(10, 201, 10))
    #to denote whether to use all the oracles of past data or not
    use_sign = [1]
    #feed_choice = 1 means feed all data before this oracle
    comb = [(gamma_value, feed_choice) for gamma_value in gamma for feed_choice in use_sign]
    return comb

def save_csv(loss_comb, fclass):
    loss_comb = loss_comb.T
    loss_comb_df = pd.DataFrame(loss_comb)
#    df2 = loss_comb_df.copy()
#    for i in list(range(int(len(loss_comb_df.columns)/2))):
#        print(i)
#        loss_comb_df[i] = df2[2*i]
#        loss_comb_df[i + 24] = df2[2*i + 1]
    loss_comb_df.to_csv(DS_RES + fclass +'.csv')

#compare the influence of changes in parameters on performance
def figure(loss_comb, fclass, did_all):
    gamma = list(range(5, 121, 5))
    for did in did_all: 
    #set false as not to feed all the previous data
        loss_comb_false = [loss_comb[2*i][did] for i in range(int(len(loss_comb)/2))]
        loss_comb_true = [loss_comb[2*i + 1][did] for i in range(int(len(loss_comb)/2))]
        #set true to feed all the previous data
        sns.set_style('darkgrid')
        plt.figure(figsize = (10, 6), dpi = 100)
        plt.plot(gamma, loss_comb_false, label = 'Data Without Previous Oracles', marker = '*')
        plt.plot(gamma, loss_comb_true, label = 'Data With Previous Oracles', marker = 'o')
        plt.xlabel("param controlling the learning rate")
        plt.title(fclass + " Fun Class for FALCON")
        plt.ylabel("Progress Validation Loss")
        plt.legend()
    
        plt.savefig(DS_RES + fclass + '_' + str(did) + '.png')
    plt.show()
    
#compared different function classes within some datasets
def compare_fclass(oracle_include =  True, fclass = ['linear', 'ridge', 'GBR']):
    df_all = []
    gamma = list(range(105, 141, 5))
    #the one without / with the previous oracles
    avg_loss_all = np.zeros((len(fclass), len(comb)))
    avg_loss_min = np.zeros(len(fclass))
    for class_index in range(len(fclass)):
        df = pd.read_csv(DS_RES + fclass[class_index] + '.csv')
        df_all.append(df)
        for i in range(len(comb)):
            avg_loss_all[class_index][i] = np.mean(list(df[str(i)]))
        avg_loss_min[class_index] = np.where(avg_loss_all[class_index] == np.min(avg_loss_all[class_index]))[0][0]
        
        print(avg_loss_min[class_index], df[str(int(avg_loss_min[class_index]))])
    #find the policy with the minimum loss in each algorthm
    
    #comparsion between different datasets
    did = 3
    sns.set_style('darkgrid')
    plt.figure(figsize = (10, 6), dpi = 100)
    for class_index in range(len(fclass)):
        plt.plot(gamma, list(df_all[class_index].loc[did])[1:25], label = fclass[class_index], marker = '*')
    plt.xlabel('param controlling the learning rate')
    plt.ylabel('Progress Validation Loss')
    plt.legend()
    plt.title('Comparison of Different Fun Class for FALCON')
    plt.savefig(DS_RES + str(did) +'_no_oracles.png')
    plt.show()
    
    plt.figure(figsize = (10, 6), dpi = 100)
    for class_index in range(len(fclass)):
        plt.plot(gamma, list(df_all[class_index].loc[did])[25:49], label = fclass[class_index], marker = '*')
    plt.xlabel('param controlling the learning rate')
    plt.ylabel('Progress Validation Loss')
    plt.legend()
    plt.title('Comparison of Different Fun Class for FALCON')
    plt.savefig(DS_RES + str(did) +'_oracles.png')
    plt.show()
    
if __name__ == '__main__':
    comb = param()
    
    print(ds_files_csv(DS_DIR), comb)
#    file_range = range(len(ds_files_csv(DS_DIR)))
#    index, pv_loss = implement_all(comb, 'cbify', 0)
#    print(pv_loss, index)
#    pv_loss_test = implement_train_test_split(comb, 'cbify')
#    print(pv_loss_test)
    
    fclass = 'linear'    
    index, loss_comb_linear = implement_all(comb, 'cbify_price', 0, None, fclass)
    #figure(loss_comb_linear, fclass, file_range)
    save_csv(loss_comb_linear, fclass)
    
    fclass = 'ridge'    
    index, loss_comb_ridge = implement_all(comb, 'cbify_price', 0, None, fclass)
    #figure(loss_comb_ridge, fclass, file_range)
    save_csv(loss_comb_ridge, fclass)
    
    fclass = 'GBR'
    index, loss_comb_GBR = implement_all(comb, 'cbify_price', 0, None,  fclass)
#    figure(loss_comb_GBR, fclass, file_range)
    save_csv(loss_comb_GBR, fclass)
    
#    compare_fclass()