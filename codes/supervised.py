#supervised algorithm for ldf
from FALCON import FALCON, FALCON_ldf
from split import sample_custom_pmf
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from glob_param import *
from base_cb import base_cb, base_cb_ldf

#use for learning_to_rank datasets.
class supervised_ldf(base_cb_ldf):
    def __init__(self, csvpath, doc_num = 10, funclass = 'linear', tau_param = 1):
        base_cb_ldf.__init__(self, csvpath, 1, doc_num, funclass, tau_param)
        
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
        model = self.offreg(0, int(self.sample_number / self.action_number))
        self.loss_all = np.zeros(self.sample_number)
        pv_loss = 0
        for epoch in range(1, self.sample_number):
            #print(self.gamma_func(epoch))
            if epoch == 1:
                for t in range(1, self.tau(1) + 1):
                    pmf = self.pmf_compute(self.context_all[self.action_number*(t - 1): self.action_number*t], model, epoch)
                    index, prob = self.sample_custom_pmf(pmf)
                    pv_loss += self.loss_encoding(index, t)
                    self.loss_all[t - 1] = pv_loss
                    #print(index, t)
                    if t == self.sample_number:
                        return pv_loss/t
                    #print(t - 1)
            if epoch > 1:
                    
                for t in range(self.tau(epoch - 1) + 1, self.tau(epoch) + 1):
                    pmf = self.pmf_compute(self.context_all[self.action_number*(t - 1): self.action_number*t], model, epoch)
                    index, prob = self.sample_custom_pmf(pmf)
                    pv_loss += self.loss_encoding(index, t)
                    self.loss_all[t - 1] = pv_loss
                    #print(index, t)
                    if t == self.sample_number:
                        return pv_loss/t
            
            
        return pv_loss/self.sample_number
    
    def pmf_compute(self, context, model, epoch):
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
        best_arm = []
        min_value = np.min(predict_y)
        for i in range(self.action_number):
            if predict_y[i] == min_value:
                best_arm.append(i)
        
        for i in best_arm:
            prob[i] = 1/len(best_arm)
        return list(prob)

if __name__ == '__main__':
    
#    for i in range(5):
#        csvpath = '../datasets/MSLR-WEB10K/ds_mslr' + str(i + 1) + '_10.csv'
#        sample_supervised = supervised_ldf(csvpath, 10, 'linear')
#        pv_loss = sample_supervised.learn_schedule()
#        print(i + 1, pv_loss)
    print("=========================")
    for i in range(2):
        csvpath = '../datasets/yahoo_dataset/set' + str(i + 1) + 'all.csv'
        sample_supervised = supervised_ldf(csvpath, 6, 'linear')
        pv_loss = sample_supervised.learn_schedule()
        print(i + 1, pv_loss)
        
    
    
        
    