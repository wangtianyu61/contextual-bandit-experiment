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

VW_DS_DIR = '../datasets/vwdatasets/'

def ds_files():
    import glob
    return sorted(glob.glob(os.path.join(VW_DS_DIR, '*.csv')))

class base_cb:
    def __init__(self, csvpath, dataset_class, funclass):
        #csvpath
        df = pd.read_csv(csvpath)
        df_feature_columns = list(set(df.columns) - {'class'} - {'Unnamed: 0'})
        
        self.context_all = self.preprocessing(df[df_feature_columns])
        
        self.context_dim = len(self.context_all)
        self.action_all = list(df['class'])
        self.action_number = len(set(self.action_all))
        #dataset_class and function class
        self.dataset_class = dataset_class
        self.funclass = funclass
        self.sample_number = len(df)
    
    #data preprocessing for later use
    def preprocessing(self, data):
        scaler = MinMaxScaler()
        scaler.fit(data)
        return scaler.transform(data)
    
    #loss encoding for evalute the performance
    def loss_encoding(self, chose_arm, true_arm):
        #print(chose_arm, true_arm)
        if chose_arm == true_arm:
            return 0
        else:
            return 1
    
    #probability to choose for which action to take
    def sample_custom_pmf(self, pmf):
        total = sum(pmf)
        scale = 1 / total
        pmf = [x * scale for x in pmf]
        draw = random.random()
        sum_prob = 0.0
        for index, prob in enumerate(pmf):
            sum_prob += prob
            if(sum_prob > draw):
                return index, prob
            
class base_cb_ldf:
    def __init__(self, csvpath, feed_choice, doc_num, funclass, tau_param):
        df = pd.read_csv(csvpath)
        self.tau_param = tau_param
        self.feed_choice = feed_choice #feed_choice = 1 means feed all data before this oracle
        feature_column = list(set(df.columns) - {'curr_num_docs', 'relevance', 'curr_arm', 'reward'})
        self.context_all = self.preprocessing(df[feature_column])
        self.action_number = doc_num
        if 'relevance' in list(df.columns):
            self.relevance_value = list(df['relevance'])
        elif 'reward' in list(df.columns):
            self.relevance_value = list(df['reward'])
        self.sample_number = int(len(df)/self.action_number)
        self.funclass = funclass
        self.context_dim = len(feature_column)
        
        
    #data preprocessing for later use
    def preprocessing(self, data):
        scaler = MinMaxScaler()
        scaler.fit(data)
        return scaler.transform(data)
            
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
        
    
     #loss encoding for evalute the performance
    def loss_encoding(self, chose_arm, t):
        #print(chose_arm, true_arm)
        if self.relevance_value[self.action_number*(t - 1) + chose_arm] in [0, 0.25, 0.5, 0.75, 1]:
            return self.relevance_value[self.action_number*(t-1) + chose_arm]
        else:
            return 0
    
    #probability to choose for which action to take
    def sample_custom_pmf(self, pmf):
        total = sum(pmf)
        scale = 1 / total
        pmf = [x * scale for x in pmf]
        draw = random.random()
        sum_prob = 0.0
        for index, prob in enumerate(pmf):
            sum_prob += prob
            if(sum_prob > draw):
                return index, prob
            
    #do the offline regression oracle for the whole schedule
    #might need some machine learning package implemented
    def data_transform(self, data_X, data_Y, index):
        data_new_X = []
        data_new_Y = []
        for i in range(len(data_X)):
            if i%self.action_number == index and data_Y[i] in [0, 0.25, 0.5, 0.75,1]:
                data_new_X.append(data_X[i])
                data_new_Y.append(data_Y[i])
        #print(index, len(data_new))
        return np.array(data_new_X), np.array(data_new_Y)
    
    def offreg(self, start, end):
        #print('offreg:', self.action_number*start, self.action_number*end)
        X = self.context_all[self.action_number*start: self.action_number*end]
        y = list(self.relevance_value[self.action_number*start: self.action_number*end])
        #loss encoding for the regression problem
        model_list = []
        for i in range(self.action_number):
            X_value_i, y_value_i = self.data_transform(X, y, i)
            y_value_i = np.array(y_value_i).T
            #print(X_value_i, y_value_i)
            if self.funclass == 'linear':
                #we first implement the linear model without constraints
                model = linear_model.LinearRegression()
            elif self.funclass == 'ridge':
                model = linear_model.Ridge(alpha = 5)
            elif self.funclass == 'GBR':
                model = ensemble.GradientBoostingRegressor(max_depth = 5, n_estimators = 200)
            
            model.fit(X_value_i, y_value_i)
            model_list.append(model)
        
        return model_list
    
    #compute the probability for sampling in every round
    #as the most relevance is 0.
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
            else:
                prob[i] = 1/(self.action_number + (- min_value + predict_y[i])*self.gamma_func(epoch))
        
        no_best_sum = np.sum(prob)
        #randomize the maximum probability
        for i in best_arm:
            prob[i] = (1 - no_best_sum)/len(best_arm)
        #print(prob)
        return list(prob)

