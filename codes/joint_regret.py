#concrete regret analysis
#two types of illustrating regret in each step
from FALCON import FALCON, FALCON_ldf, FALCON_price
from split_gz_csv import ds_files_csv, ds_files
from split import sample_custom_pmf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from glob_param import *
from run_vw_job import process
from split import *
import re
rgx = re.compile('^average loss = (.*)$', flags=re.M)


#define a function ensuring that the vw benchmark method could be applied
def regret_figure(loss_all, alg_name, did, did_type):

    plt.figure(figsize = (10, 6), dpi = 100)
    sns.set_style('darkgrid')

    for i in range(len(loss_all)):
        plt.plot(list(range(len(loss_all[i]))), loss_all[i], label = alg_name[i])
    
    plt.xlabel('Number of Customers')
    plt.ylabel('Cummulative Revenue')
    plt.legend()
    plt.savefig('regret_figure/' + did_type + '_' + str(did) + '.png')
    plt.show()

#transform the output value into the loss in each round
def output_extraction(output):
    output_all = output.split('\n')
    output_loss_epoch = []
    for index in range(len(output_all)):
        #print(output_all[index][0])
        if len(output_all[index].split(' ')) in [36, 38, 40, 42, 44]:
            output_loss_epoch.append(float(output_all[index].split(' ')[0]))
    
    cdf_loss = np.multiply(output_loss_epoch, list(range(1, len(output_loss_epoch) + 1)))
    return cdf_loss

#list the param for the vw format for each algorithm in ranking dta
def rank_param_vw():
    alg_param = []
    #greedy alg
    alg_param.append({'alg':('epsilon', 0), 'learning_rate':1.0, 'cb_type':'mtr'})
    #epsilon_greedy alg
    alg_param.append({'alg':('epsilon', 0.02), 'learning_rate':3.0, 'cb_type':'mtr'})
    #bag_algorithm
    alg_param.append({'alg':('bag', 4), 'learning_rate':1.0, 'cb_type':'mtr'})
    #bag greedy algorithm
    alg_param.append({'alg':('bag', 16, 'greedify', None), 'learning_rate':1.0, 'cb_type':'mtr'})
    #cover
    alg_param.append({'alg':('cover', 1), 'learning_rate':3.0, 'cb_type':'mtr'})
    #cover_nu
    alg_param.append({'alg':('cover', 1, 'nounif', None), 'learning_rate':1.0, 'cb_type':'mtr'})
    
    return alg_param

#list the param for the vw format for each algorithm in openml dta
def openml_param_vw():
    alg_param = []
    #greedy alg
    alg_param.append({'alg':('epsilon', 0), 'learning_rate':3.0, 'cb_type':'dr'})
    #epsilon_greedy alg
    alg_param.append({'alg':('epsilon', 0.05), 'learning_rate':10.0, 'cb_type':'mtr'})
    #bag_algorithm
    alg_param.append({'alg':('bag', 4), 'learning_rate':10.0, 'cb_type':'mtr'})
    #bag greedy algorithm
    alg_param.append({'alg':('bag', 2, 'greedify', None), 'learning_rate':3.0, 'cb_type':'dr'})
    #cover
    alg_param.append({'alg':('cover', 1), 'learning_rate':3.0, 'cb_type':'mtr'})
    #cover_nu
    alg_param.append({'alg':('cover', 16, 'psi', 0, 'nounif', None), 'learning_rate':10.0, 'cb_type':'dr'})
    return alg_param


#list the param for the vw format for each algorithm in loan data
def loan_param_vw():
    alg_param = []
    #greedy alg
    alg_param.append({'alg':('epsilon', 0), 'learning_rate':0.001, 'cb_type':'dr'})
    #epsilon_greedy alg
    alg_param.append({'alg':('epsilon', 0.02), 'learning_rate':0.3, 'cb_type':'ips'})
    #bag_algorithm
    alg_param.append({'alg':('bag', 4), 'learning_rate':0.1, 'cb_type':'mtr'})
    #bag greedy algorithm
    alg_param.append({'alg':('bag', 16, 'greedify', None), 'learning_rate':0.001, 'cb_type':'dr'})
    #cover
    alg_param.append({'alg':('cover', 16), 'learning_rate':0.001, 'cb_type':'ips'})
    #cover_nu
    alg_param.append({'alg':('cover', 8, 'psi', 1.0, 'nounif', None), 'learning_rate':0.3, 'cb_type':'dr'})
    return alg_param
  



#show the regret performance in each algorithm
def alg_performance_regret(DS_DIR, did, alg_param, did_type):
    loss_param = []
    csv_path = ds_files_csv(DS_DIR)[did]
    gz_path = ds_files(DS_DIR)[did]
    if did_type == 'openml':
        sample_falcon = FALCON(csv_path, 95, 1)
    elif did_type == 'ms':
        sample_falcon = FALCON_ldf(csv_path, 105, 1, 10)
    elif did_type == 'yahoo':
        sample_falcon = FALCON_ldf(csv_path, 105, 1, 6, 'ridge')
    elif did_type == 'loan':
        if did == 0:
            sample_falcon = FALCON_price(csv_path, 10, 1, 15)
        elif did == 1:
            sample_falcon = FALCON_price(csv_path, 10, 1, 10)
        
    sample_falcon.learn_schedule()
    loss_param.append(sample_falcon.loss_reward)
        
    gz_path = ds_files(DS_DIR)[did]
    for index in range(len(alg_param)):
        pv_loss, output = process(gz_path, alg_param[index], None, True, did_type)
        loss_param.append(list(output_extraction(output)))
    return loss_param

#show the function effects for FALCON in loan data set
def alg_performance_regret_FALCON(DS_DIR):
    loss_param = []
    csv_path = ds_files_csv(DS_DIR)
    did_price_range = [15, 10]
    
    for did in range(2):
        #linear
        sample_falcon = FALCON_price(csv_path[did], 10, 1, did_price_range[did], 'linear')
        sample_falcon.learn_schedule()
        loss_param.append(sample_falcon.loss_reward)
        
        #ridge
        sample_falcon = FALCON_price(csv_path[did], 50, 1, did_price_range[did], 'ridge')
        sample_falcon.learn_schedule()
        loss_param.append(sample_falcon.loss_reward)
        
        #GBR
        sample_falcon = FALCON_price(csv_path[did], 140, 1, did_price_range[did], 'GBR')
        sample_falcon.learn_schedule()
        loss_param.append(sample_falcon.loss_reward)
      
    loss_param.append(sample_falcon.loss_reward)
    return loss_param

#show the performance in each data set class
def dta_oracle(DS_DIR, alg_param, did_type):
    for did in range(len(ds_files(DS_DIR))):    
        loss_param = alg_performance_regret(DS_DIR, did, alg_param, did_type)
        regret_figure(loss_param, alg_name, did, did_type)
        
if __name__ == '__main__':
    
    alg_name = ['FALCON', 'greedy','epsilon-greedy', 'bag', 'bag-greedy', 'cover', 'cover-nu']
#    alg_param = openml_param_vw()
#    
#    did_type = 'openml'
#    DS_DIR = '../datasets/vwdatasets/'
#    dta_oracle(DS_DIR, alg_param, did_type)
    
#    alg_param = rank_param_vw()    
#    did_type = 'ms'
#    DS_DIR = '../datasets/MSLR-WEB10K/'
#    dta_oracle(DS_DIR, alg_param, did_type)
#    
#    did_type = 'yahoo'
#    DS_DIR = '../datasets/yahoo_dataset/'
#    dta_oracle(DS_DIR, alg_param, did_type)
    alg_param = loan_param_vw()
    DS_DIR = '../datasets/OnlineAutoLoan/'
    dta_oracle(DS_DIR, alg_param, 'loan')
