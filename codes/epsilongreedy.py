import argparse
import numpy as np
import pandas as pd
import gzip
import os
import random
import scipy.sparse as sp
import time
import math
from sklearn import linear_model, ensemble
from sklearn.preprocessing import MinMaxScaler
from base_cb import base_cb, base_cb_ldf
from CSLR import *

VW_DS_DIR = '../datasets/vwdatasets/'

def ds_files():
    import glob
    return sorted(glob.glob(os.path.join(VW_DS_DIR, '*.csv')))


class epsilon_greedy(base_cb):
    #baseline model
    def learn_schedule_type0(self, epsilon):
        self.epsilon = epsilon
        pv_loss = 0
        for t in range(1, self.sample_number + 1):
            pmf = self.pmf_compute(self.context_all[t - 1: t], self.baseline_model)
            index, prob = self.sample_custom_pmf(pmf)
            pv_loss += self.loss_encoding(index, self.action_all[t - 1])
        return pv_loss/self.sample_number
    
    #uniformly search and then commit to the best policy
    ## split_point controls the split between exploration and exploitation
    def learn_schedule_type1(self, split_point = None):
        if split_point == None:
            split_point = int(math.pow(self.sample_number, 2/3))
        else:
            split_point = int(math.pow(self.sample_number, split_point))
        pv_loss = 0
        for t in range(1, split_point):
            index, prob = self.sample_custom_pmf(list(np.ones(self.action_number)))
            pv_loss += self.loss_encoding(index, self.action_all[t - 1])
            #print(t)
        model = self.offreg(0, split_point - 1)
        for t in range(split_point, self.sample_number + 1):
            pmf = self.pmf_compute(self.context_all[t - 1: t], model, 0)
            index, prob = self.sample_custom_pmf(pmf)
            pv_loss += self.loss_encoding(index, self.action_all[t - 1])
            #print(t)
        return pv_loss/self.sample_number
            
    #Langford and Zhang 2008 epoch-greedy algorithm
    def epoch_schedule(self, epsilon):
        pv_loss = 0
        temp_model = None
        for epoch in range(1, self.sample_number):
            #print(self.gamma_func(epoch))
            if epoch == 1:
                for t in range(1, self.tau(1) + 1):
                    pmf = list(np.ones(self.action_number))
                    index, prob = self.sample_custom_pmf(pmf)
                    pv_loss += self.loss_encoding(index, self.action_all[t - 1])
                    if t == self.sample_number:
                        return pv_loss/t
                    #print(t - 1)
            if epoch > 1:
                #in case of no online data last epoch 
                if self.tau(epoch - 1) - self.tau(epoch - 2) > 0:
                    if self.feed_choice == 0:
                        start = self.tau(epoch - 2)
                    else:
                        start = 0
                    end = self.tau(epoch - 1)
                
                    model = self.offreg(start, end)
                    temp_model = model
                else:
                    model = temp_model
                for t in range(self.tau(epoch - 1) + 1, self.tau(epoch) + 1):
                    pmf = self.pmf_compute(self.context_all[t - 1: t], model, epsilon)
                    index, prob = self.sample_custom_pmf(pmf)
                    pv_loss += self.loss_encoding(index, self.action_all[t - 1])
                    #print(t - 1)
                    if t == self.sample_number:
                        return pv_loss/t
        return pv_loss/self.sample_number


        
class epsilongreedy_ldf(base_cb_ldf):
    def __init__(self, csvpath, feed_choice = 0, doc_num = 10, funclass = 'linear', tau_param = 1):
        
        base_cb_ldf.__init__(self, csvpath, feed_choice, doc_num, funclass, tau_param)
    #the overall schedule for implementing that algorithm
    ##pay attention to that the index for dataset and the time do not match 
    def learn_schedule(self, epsilon):
        self.loss_all = np.zeros(self.sample_number)
        pv_loss = 0
        temp_model = None
        for epoch in range(1, self.sample_number):
            #print(self.gamma_func(epoch))
            if epoch == 1:
                for t in range(1, self.tau(1) + 1):
                    pmf = list(np.ones(self.action_number))
                    index, prob = self.sample_custom_pmf(pmf)
                    pv_loss += self.loss_encoding(index, t)
                    self.loss_all[t - 1] = pv_loss
                    #print(index, t)
                    if t == self.sample_number:
                        return pv_loss/t
                    #print(t - 1)
            if epoch > 1:
                #in case of no online data last epoch 
                if self.tau(epoch - 1) - self.tau(epoch - 2) > 0:
                    if self.feed_choice == 0:
                        start = self.tau(epoch - 2)
                    else:
                        start = 0
                    end = self.tau(epoch - 1)
                    model = self.offreg(start, end)
                    temp_model = model
                else:
                    model = temp_model
                    
                for t in range(self.tau(epoch - 1) + 1, self.tau(epoch) + 1):
                    pmf = self.pmf_compute(self.context_all[self.action_number*(t - 1): self.action_number*t], model, epsilon)
                    index, prob = self.sample_custom_pmf(pmf)
                    pv_loss += self.loss_encoding(index, t)
                    self.loss_all[t - 1] = pv_loss
                    #print(index, t)
                    if t == self.sample_number:
                        return pv_loss/t
            
            
        return pv_loss/self.sample_number

    #compute the probability for sampling in every round
    def pmf_compute(self, context, model, epsilon = None):
        context = np.array(context)
        predict_y = np.zeros(self.action_number)
        prob = np.zeros(self.action_number)
        for i in range(self.action_number):
            if context[i][0] >= 0:
                #a large number preventing it from being selected
                predict_y[i] = model[i].predict(np.array([context[i]]))
            else:
                predict_y[i] = 10000
        #print(predict_y)
        if epsilon != None:
            self.epsilon = epsilon
        best_arm = []
        min_value = np.min(predict_y)
        for i in range(self.action_number):
            if predict_y[i] == min_value:
                best_arm.append(i)
        draw = random.random()
        #uniformly choose
        if draw <= self.epsilon:
            return list(np.ones(self.action_number))
        else:
            #randomize the maximum probability
            for i in best_arm:
                prob[i] = 1/len(best_arm)
            #print(prob)
            return prob
if __name__ == '__main__':
    epsilon_list = [0]
    pv_loss = np.zeros((7, len(epsilon_list)))
    for i in range(3, 5):
        csvpath = '../datasets/MSLR-WEB10K/ds_mslr' + str(i + 1) + '_10.csv'
        sample_falcon = epsilongreedy_ldf(csvpath, 1, 10, 'GBR', 1)
        for j in range(len(epsilon_list)):    
            pv_loss[i][j] = sample_falcon.learn_schedule(epsilon_list[j])
#    for i in range(2):
#        csvpath = '../datasets/yahoo_dataset/set' + str(i + 1) + 'all.csv'
#        sample_falcon = epsilongreedy_ldf(csvpath, 1, 6, 'GBR')
#        for j in range(len(epsilon_list)):
#            pv_loss[5 + i][j] = sample_falcon.learn_schedule(epsilon_list[j])
#    
    
    