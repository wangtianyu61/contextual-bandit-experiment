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
from base_cb import base_cb, base_cb_ldf
from CSLR import *
from sklearn.preprocessing import MinMaxScaler

VW_DS_DIR = '../datasets/vwdatasets/'
HOTEL_DS_DIR = '../datasets/pricing/MSOM_Hotel/'

def ds_files(DS_DIR):
    import glob
    return sorted(glob.glob(os.path.join(DS_DIR, '*.csv')))

class FALCON(base_cb):
    def __init__(self, csvpath, gamma_param, feed_choice = 0, funclass = 'linear', tau_param = 1, fun_constr = False, dataset_class = 'oml'):
        
        self.tau_param = tau_param
        self.gamma_param = gamma_param
        self.feed_choice = feed_choice #feed_choice = 1 means feed all data before this oracle
        self.fun_constr = fun_constr
        base_cb.__init__(self, csvpath, dataset_class, funclass)
    
    #parameter for sampling
    def gamma_func(self, epoch):
        if epoch == 1:
            return 1
        else:
            return self.gamma_param*math.sqrt(self.action_number*(self.tau(epoch - 1) - self.tau(epoch - 2)*(1 - self.feed_choice))/self.context_dim)
    
    #epoch round
    def tau(self, epoch):
        if epoch == 0:
            return 0
        elif self.tau_param == 1:
            return int(math.pow(2, epoch))
        elif self.tau_param == 2:
            return int(math.pow(self.sample_number, 1 - math.pow(0.5, epoch)))
        
    #the overall schedule for implementing that algorithm
    ##pay attention to that the index for dataset and the time do not match 
    def learn_schedule(self):
        self.loss_all = np.zeros(self.sample_number)
        pv_loss = 0
        temp_model = None
        for epoch in range(1, self.sample_number):
            #print(self.gamma_func(epoch))
            if epoch == 1:
                print(self.tau(1) + 1)
                for t in range(1, self.tau(1) + 1):
                    pmf = list(np.ones(self.action_number))
                    index, prob = self.sample_custom_pmf(pmf)
                    pv_loss += self.loss_encoding(index, self.action_all[t - 1])
                    self.loss_all[t - 1] = pv_loss
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
                    pmf = self.pmf_compute(self.context_all[t - 1: t], model, epoch)
                    index, prob = self.sample_custom_pmf(pmf)
                    pv_loss += self.loss_encoding(index, self.action_all[t - 1])
                    self.loss_all[t - 1] = pv_loss
                    #print(t - 1)
                    if t == self.sample_number:
                        return pv_loss/t
            
            
        return pv_loss/self.sample_number
    
    #do the offline regression oracle for the whole schedule
    #might need some machine learning package implemented
    def offreg(self, start, end):
        X = self.context_all[start: end]
        y = self.action_all[start: end]
        #loss encoding for the regression problem
        model_list = []
        for i in range(self.action_number):
            y_value_i = [1 if index == i else 0 for index in y]    
            y_value_i = np.array(y_value_i).T
            
            if self.funclass == 'linear':
                #we first implement the linear model without constraints
                if self.fun_constr == True:
                    model = ConstrainedLinearRegression()
                else:
                    model = linear_model.LinearRegression()
            elif self.funclass == 'ridge':
                model = linear_model.Ridge(alpha = 1)
            elif self.funclass == 'GBR':
                model = ensemble.GradientBoostingRegressor(max_depth = 5, n_estimators = 100)
            model.fit(X, y_value_i)
            model_list.append(model)
        
        return model_list
    
    #compute the probability for sampling in every round
    def pmf_compute(self, context, model, epoch):
        context = np.array(context)
        predict_y = np.zeros(self.action_number)
        prob = np.zeros(self.action_number)
        for i in range(self.action_number):
            predict_y[i] = model[i].predict(context)
        #print(predict_y)
        best_arm = []
        max_value = np.max(predict_y)
        for i in range(self.action_number):
            if predict_y[i] == max_value:
                best_arm.append(i)
            else:
                prob[i] = 1/(self.action_number + (max_value - predict_y[i])*self.gamma_func(epoch))
        
        no_best_sum = np.sum(prob)
        #randomize the maximum probability
        for i in best_arm:
            prob[i] = (1 - no_best_sum)/len(best_arm)
        #print(prob)
        return list(prob)
    
#use for learning_to_rank datasets.
class FALCON_ldf(base_cb_ldf):
    def __init__(self, csvpath, gamma_param, feed_choice = 0, doc_num = 10, funclass = 'linear', tau_param = 1):
        self.gamma_param = gamma_param
        base_cb_ldf.__init__(self, csvpath, feed_choice, doc_num, funclass, tau_param)
        
    #parameter for sampling
    def gamma_func(self, epoch):
        if epoch == 1:
            return 1
        else:
            return self.gamma_param*math.sqrt(self.action_number*(self.tau(epoch - 1) - self.tau(epoch - 2)*(1 - self.feed_choice))/self.context_dim)
    
    #epoch round
    def tau(self, epoch):
        if epoch == 0:
            return 0
        elif self.tau_param == 1:
            return int(math.pow(2, epoch))
        elif self.tau_param == 2:
            return int(math.pow(self.sample_number, 1 - math.pow(0.5, epoch)))
    
    #the overall schedule for implementing that algorithm
    ##pay attention to that the index for dataset and the time do not match 
    def learn_schedule(self):
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
                    pmf = self.pmf_compute(self.context_all[self.action_number*(t - 1): self.action_number*t], model, epoch)
                    index, prob = self.sample_custom_pmf(pmf)
                    pv_loss += self.loss_encoding(index, t)
                    self.loss_all[t - 1] = pv_loss
                    #print(index, t)
                    if t == self.sample_number:
                        return pv_loss/t
            
            
        return pv_loss/self.sample_number
    
    
#use for learning_to_rank datasets.
class FALCON_price(base_cb_ldf):
    def __init__(self, csvpath, gamma_param, feed_choice = 0, doc_num = 10, funclass = 'linear', tau_param = 1):
        self.gamma_param = gamma_param
        base_cb_ldf.__init__(self, csvpath, feed_choice, doc_num, funclass, tau_param)
        
   
    #loss encoding for evalute the performance
    def loss_encoding(self, chose_arm, t):
        return self.relevance_value[(t - 1)*self.action_number + chose_arm]
    
    #the overall schedule for implementing that algorithm
    ##pay attention to that the index for dataset and the time do not match 
    def learn_schedule(self):
        self.loss_reward = np.zeros(self.sample_number)
        self.chose_arm = []
        pv_loss = 0
        temp_model = None
        for epoch in range(1, self.sample_number):
            #print(self.gamma_func(epoch))
            if epoch == 1:
                for t in range(1, self.tau(1) + 1):
                    pmf = list(np.ones(self.action_number))
                    index, prob = self.sample_custom_pmf(pmf)
                    self.chose_arm.append(index)
                    pv_loss += self.loss_encoding(index, t)
                    self.loss_reward[t - 1] = pv_loss
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
                    pmf = self.pmf_compute(self.context_all[self.action_number*(t - 1): self.action_number*t], model, epoch)
                    index, prob = self.sample_custom_pmf(pmf)
                    #print(pmf)
                    self.chose_arm.append(index)
                    pv_loss += self.loss_encoding(index, t)
                    self.loss_reward[t - 1] = pv_loss
                    #print(index, t)
                    if t == self.sample_number:
                        return pv_loss/t
            #print(pv_loss)
            
        return pv_loss/self.sample_number


    def offreg(self, start, end):
        #print('offreg:', self.action_number*start, self.action_number*end)
        X = np.zeros((end - start, self.context_all.shape[1]))
        y = np.zeros((end - start, 1))
        #print(start, end)
        for t in range(start + 1, end + 1):
            #print(t)
        #loss encoding for the regression problem
            X[t - start - 1] = self.context_all[(t - 1)*self.action_number + self.chose_arm[t - 1]]
            
            y[t - start - 1] = np.array([self.relevance_value[(t - 1)*self.action_number + self.chose_arm[t - 1]]])
        
        y = y.ravel()
        
        if self.funclass == 'linear':
            #we first implement the linear model without constraints
            model = linear_model.LinearRegression()
        elif self.funclass == 'ridge':
            model = linear_model.Ridge(alpha = 1)
        elif self.funclass == 'GBR':
            model = ensemble.GradientBoostingRegressor(max_depth = 5, n_estimators = 100)
            
        model.fit(X, y)
        
        return model
    
    #compute the probability for sampling in every round
    #as the most relevance is 0.
    def pmf_compute(self, context, model, epoch):
        context = np.array(context)
        predict_y = np.zeros(self.action_number)
        prob = np.zeros(self.action_number)
        for i in range(self.action_number):
            #a large number preventing it from being selected
            predict_y[i] = model.predict(np.array([context[i]]))

        #print(predict_y)
        best_arm = []
        max_value = np.max(predict_y)
        for i in range(self.action_number):
            if predict_y[i] == max_value:
                best_arm.append(i)
            else:
                prob[i] = 1/(self.action_number + (max_value - predict_y[i])*self.gamma_func(epoch))
        
        no_best_sum = np.sum(prob)
        #randomize the maximum probability
        for i in best_arm:
            prob[i] = (1 - no_best_sum)/len(best_arm)
        #print(prob)
        return list(prob)
    
if __name__ == '__main__':
#    pv_loss = np.zeros((7, 9))
#    for i in range(5):
#        csvpath = '../datasets/MSLR-WEB10K/ds_mslr' + str(i + 1) + '_10.csv'
#        for gamma in range(80, 86, 5):
#            sample_falcon = FALCON_ldf(csvpath, gamma, 1, 10, 'linear')
#            pv_loss[i][int(gamma/5)-16] = sample_falcon.learn_schedule()

    csvpath = '../datasets/OnlineAutoLoan/ds_loan1_15.csv'
    for gamma in [50]:
        sample_falcon = FALCON_price(csvpath, gamma, 1, 10, 'linear')
        print(sample_falcon.learn_schedule())
        
#    for i in range(2):
#        csvpath = '../datasets/yahoo_dataset/set' + str(i + 1) + 'all.csv'
#        for gamma in range(80, 121, 5):
#            sample_falcon = FALCON_ldf(csvpath, gamma, 1, 6, 'GBR')
#            pv_loss[5 + i][int(gamma/5)-16] = sample_falcon.learn_schedule()
        
#    for gamma in range(5,141,5):
#        sample = FALCON(ds_files(HOTEL_DS_DIR)[0], gamma, 1, 'linear')
#        pv_loss = sample.learn_schedule()
#        print(gamma, pv_loss)
#    for i in range(9):
#        print("===========================\n", i)
#        t = time.time()
#        sample = FALCON(ds_files()[i], 50, 0, 'linear')
#        pv_loss = sample.learn_schedule()
#        print('pv_loss: ', pv_loss, 'elasped time: ', time.time() - t)
#        sample = FALCON(ds_files()[i], 50, 0, 'ridge')
#        pv_loss = sample.learn_schedule()
#        print('pv_loss: ', pv_loss, 'elasped time: ', time.time() - t)
#        sample = FALCON(ds_files()[i], 50, 0, 'linear')
#        pv_loss = sample.learn_schedule()
#        print('pv_loss: ', pv_loss, 'elasped time: ', time.time() - t)
    #did = 4
    ##import matplotlib.pyplot as plt
    #loss_list = []
    #print("Not use all dataset of previous oracle constraint:")
    #for gamma in range(5, 81, 5):
    #    t = time.time()
    #    sample_falcon = FALCON(ds_files()[did], gamma, 0)
    #    pv_loss = sample_falcon.learn_schedule()
    #    loss_list.append(pv_loss)
    #    print ('gamma:', gamma, 'elapsed time:', time.time() - t, 'pv loss:', pv_loss)

    #plt.plot(list(range(5, 81, 5)), loss_list, label = 'pv_loss', marker = 'o')
    #
    #plt.legend()
    #
    #plt.xlabel('gamma')
    #plt.ylabel('pv_loss')
    #
    #plt.show()
    #print("==================\nimpose the parameter constraint:")
    #for gamma in range(5, 51, 5):
    #    t = time.time()
    #    sample_falcon = FALCON(ds_files()[did], gamma, 0, 1, True)
    #    pv_loss = sample_falcon.learn_schedule()
    #    print ('gamma:', gamma, 'elapsed time:', time.time() - t, 'pv loss:', pv_loss)
    
