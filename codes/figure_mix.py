import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from glob_param import *

#transform the list into 0-1 interval
def normalize_list(did_list):
    Max = np.max(did_list)
    Min = np.min(did_list)
    return [(value - Min)/(Max - Min) for value in did_list]
#transform the data into the 0-1 interval
def normalize_df(df, include_supervised):
    openml_did = list(list(df.columns)[2:])
    for did in openml_did:
        #whether to include the algorithm of supervised learning
        if include_supervised == True:
            loss_did = normalize_list(list(df[did]))
            df[did] = pd.Series(loss_did)  
        else:
            loss_did = normalize_list(list(df[did])[1:])
            df[did] = pd.Series([df.loc[0, did]] + loss_did)
            
            

    return df        

def aggregate_figure(df, include_supervised):
    if include_supervised == False:
        include_supervised = 1
    else:
        include_supervised = 0
    pv_loss_all = []
    alg_label = list(df['did'])[include_supervised:]
    
    did_num = len(df.columns[2:])
    did_list = list(range(0, did_num + 1)) + [did_num]
    plt.figure(figsize = (10.5, 6.5), dpi = 100)
    sns.set_style('darkgrid')
    for i in range(include_supervised , len(df)):
        norm_pv_loss = [0] + sorted(list(df.loc[i])[2:]) + [1]
        print(norm_pv_loss)
        pv_loss_all.append(norm_pv_loss)
        #print(norm_pv_loss)
        if alg_label[i - include_supervised] != 'FALCON':
            plt.step(did_list, norm_pv_loss, where = 'pre', label = alg_label[i - include_supervised], linewidth = '1')
        else:
            plt.step(did_list, norm_pv_loss, where = 'pre', label = alg_label[i - include_supervised], linewidth = '3')
    
    plt.xlabel('Number of datasets')
    plt.ylabel('Normalized relative loss')
    
    
    pyplot.xticks(range(did_num + 1))
    plt.legend(loc = 2)
    plt.show()

    


def figure(loss_comb, fclass, did_all):
    gamma = list(range(5, 101, 5))
    for did in did_all: 
    #set false as not to feed all the previous data
        loss_comb_false = [loss_comb[2*i][did] for i in range(int(len(loss_comb)/2))]
        loss_comb_true = [loss_comb[2*i + 1][did] for i in range(int(len(loss_comb)/2))]
        #set true to feed all the previous data
        sns.set_style('dark')
        plt.figure(figsize = (10, 6), dpi = 100)
        plt.plot(gamma, loss_comb_false, label = 'Data Without Previous Oracles', marker = '*')
        plt.plot(gamma, loss_comb_true, label = 'Data With Previous Oracles', marker = 'o')
        plt.xlabel("param controlling the learning rate")
        plt.title(fclass + " Fun Class for FALCON")
        plt.ylabel("Progress Validation Loss")
        plt.legend(loc = 1)
    
        plt.savefig(DS_RES + fclass + '_' + str(did) + '.png')
    
    plt.show()
    
if __name__ == '__main__':
    #split_percent == 1 means using 100% of the datasets and then compare performance between hyperparams
    split_percent = ''
    include_supervised = False
    df = pd.read_csv('vw_rank/' + 'avg_test' + str(split_percent) + '.csv')
    df = normalize_df(df, include_supervised)
    aggregate_figure(df, include_supervised)
#    df1 = pd.read_csv(VW_DS_RES + 'ridge.csv')
#    df2 = df1.copy()
#    for i in list(df1.columns)[0:20]:
#        df1[i] = df2[str(2*int(i))]
#        df1[str(int(i) + 20)] = df2[str(2*int(i) + 1)]
#    df1.to_csv(VW_DS_RES + 'ridge1.csv')

    