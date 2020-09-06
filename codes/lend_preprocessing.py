#preprocessing the data from online lending
import pandas as pd
import numpy as np
import gzip, csv
import os
import scipy.sparse as sp
import math
import random
from sklearn.linear_model import LogisticRegression, LinearRegression

VW_DS_DIR = '../datasets/OnlineAutoLoan/'
# encode the original data into vw format
def save_vw_dataset(X, y, did, ds_dir, choice = True):
    # change a bit as we change the data structure of y
    n_classes = max(y) + 1 
    print("check step")
    fname = 'ds_loan{}_{}'.format(did, n_classes)
    #add it to the csv path for the use of bypassing the monster
    
    
     
#    with gzip.open(os.path.join(ds_dir, fname + '.vw.gz'), 'w') as f:
#        #if the matrix is sparse then we use the nonzero value to fill.
#        if sp.isspmatrix_csr(X):
#            for i in range(X.shape[0]):
#                str_output = '{} | {}\n'.format(y[i] + 1, ' '.join(
#                    '{}:{:.6f}'.format(j, val) for j, val in zip(list(X.loc[i].index), list(X.loc[i]))))
#                f.write(str_output.encode())
#        else:
#            for i in range(X.shape[0]):
#                print('yes')
#                str_output = '{} | {}\n'.format(y[i] + 1, ' '.join(
#                    '{}:{:.6f}'.format(j, val) for j, val in enumerate(X.loc[i])))
#                f.write(str_output.encode())
#    if choice == False:
#        return
    X['class'] = pd.Series(y)
    X.to_csv(VW_DS_DIR + fname + '.csv', index = None)
# map the xvalue and yvalue into discrete type
def map_class(X):
    
    ## map x
    for feature in ['CarType']:
        feature_type = list(set(list(X[feature])))
        feature_type = list(pd.Series(feature_type).dropna())
        feature_number = len(feature_type)
        feature_label = []
        feature_map = dict(zip(feature_type, list(range(1, feature_number + 1))))
        for item in list(X[feature]):
            #map the null value into zero
            if item == '' or item == None or (type(item) == float and np.isnan(item) == True):
                feature_label.append(0)
            else:
                feature_label.append(feature_map[item])
        X[feature] = pd.Series(feature_label)
    return X

#shuffle the value of x and y to a stochastic order
def shuffle(X, y):
    n = X.shape[0]
    perm = np.random.permutation(n)
    #print(n, perm)
    X_shuf = X.loc[perm]
    y_shuf = y[perm]
    return X_shuf, y_shuf

def preprocessing():
    df = pd.read_csv(VW_DS_DIR + 'Original/Data.csv')
    feature_column = ['Tier', 'State', 'termclass', 'Type', 'CarType', 'partnerbin', 'Primary_FICO', 'Term', 'rate', 'Competition_rate',
                      'onemonth', 'rate1', 'rel_compet_rate', 'mp', 'mp_rto_amtfinance', 'Amount_Approved']
    decision_column = ['apply']
    item_number = 50000
    for i in range(4):
        df_temp = df[feature_column + decision_column].loc[item_number*i: item_number*(i + 1)]
        df_temp.to_csv(VW_DS_DIR + 'loan' + str(i + 1) + '.csv', index = None)

def change_vw(ds_dir, fname, X, y):
    with gzip.open(os.path.join(ds_dir, fname + '.vw.gz'), 'w') as f:
    #if the matrix is sparse then we use the nonzero value to fill.
        if sp.isspmatrix_csr(X):
            for i in range(X.shape[0]):
                str_output = '{} | {}\n'.format(y[i] + 1, ' '.join(
                        '{}:{:.6f}'.format(j, val) for j, val in zip(list(X.loc[i].index), list(X.loc[i]))))
                f.write(str_output.encode())
        else:
            for i in range(X.shape[0]):
                print('yes')
                str_output = '{} | {}\n'.format(y[i] + 1, ' '.join(
                '{}:{:.6f}'.format(j, val) for j, val in enumerate(X.loc[i])))
                f.write(str_output.encode())

def price_compute(MP, Rate, Term, LoanAmount):
    #average monthly LIBOR
    #LIBOR = math.pow(1.0177, 1/12) - 1
    LIBOR = np.array(Rate)*0.0001 
    price = -1*np.array(LoanAmount)
    for i in range(len(Rate)):
        for j in range(1, Term[i] + 1):
            price[i] += MP[i]/((1 + LIBOR[i])**j)
    print(price)
    return price

def feature_normalize(feature_column, price):
    X = map_class(df[feature_column])
    for column in X.columns:
        X[column] = pd.Series(np.array(list(X[column]))/np.mean(X[column]))
    print(X)
    Y = np.zeros(X.shape)
    for index in range(len(X)):
        
        Y[index] = np.array(list(X.loc[index]))*price[index]
    comb = pd.concat([X, pd.DataFrame(Y)], axis = 1)
    return X, comb
    



price_arm = 10
price_range = 1000

def simulation_data_generation(lr_model, X_original, price):

    noise_std = 0.004
    
    csv_loan = open(VW_DS_DIR + str(price_arm) + '_' + str(price_range) + '.csv', 'w', newline = '')
    
    csv_writer = csv.writer(csv_loan)
    
    csv_writer.writerow(['curr_arm', 'reward'] + list(range(1, 2*len(X_original.columns) + 1)))
    
    
    arm_price = np.array(list(range(1, price_arm + 1)))*price_range
    Y = np.zeros(X_original.shape[1])
    reward = np.zeros((X_original.shape[0], price_arm))
    
    for t in range(X_original.shape[0]):
        for i in range(price_arm):
            Y = np.array(list(X_original.loc[t]))*arm_price[i]
            X_temp = X_original.loc[t:t]
            X_temp.index = [0]
            comb = pd.concat([X_temp, pd.DataFrame(Y).T], axis = 1)
            #print(comb)
            prob = 1/(1 + math.exp(np.dot(-1*lr_model.coef_[0], np.array(list(comb.loc[0]))))) + np.random.normal(0, noise_std)
            if prob >= random.random():
                reward[t][i] = arm_price[i]
            csv_writer.writerow([i + 1, reward[t][i]] + list(comb.loc[0]))
        
    csv_loan.close()
    return reward

#convert the csv format to the gz for vw
def csv_to_gz():
    df = pd.read_csv(VW_DS_DIR + str(price_arm) +'_' + str(price_range) + '.csv')
    feature_column = list(df.columns)[2:]
    X_feature = df[feature_column]
    curr_arm = list(df['curr_arm'])
    reward = list(df['reward'])
    fname = 'ds_{}_{}'.format('loan2', price_arm)
    #add it to the csv path for the use of bypassing the monster
    with gzip.open(os.path.join(VW_DS_DIR, fname + '.vw.gz'), 'w') as f:
        #if the matrix is sparse then we use the nonzero value to fill.
        for i in range(X_feature.shape[0]):
            str_output = '{}:{} | {}\n'.format(curr_arm[i], reward[i], ' '.join(
                    '{}:{:.3f}'.format(j, val) for j, val in zip(list(feature_column), list(X_feature.loc[i]))))
            f.write(str_output.encode())
            if curr_arm[i] == price_arm:
                f.write("\n".encode())
    
    
    
if __name__ == '__main__':
#    preprocessing()
#    for i in range(4):
#        df_temp = pd.read_csv(VW_DS_DIR + 'loan' + str(i + 1) + '.csv') 
#        X, y = shuffle(df_temp[list(df_temp.columns)[:-1]], df_temp['apply'])
#        X = map_class(X)
#        save_vw_dataset(X, list(y), i + 1, VW_DS_DIR)
    
#    feature_column = ['intercept', 'CarType', 'Primary_FICO', 'Term', 'rate', 'Competition_rate',
 #                     'onemonth',  'Amount_Approved']
#    feature_column = ['intercept', 'CarType', 'Primary_FICO', 'Term', 'Competition_rate', 'onemonth']
#    df = pd.read_csv(VW_DS_DIR + 'Original/Data.csv')[0:20]
#    df['intercept'] = pd.Series(list(np.ones(len(df))))
#    price = price_compute(list(df['mp']), list(df['rate']), list(df['Term']), list(df['Amount_Approved']))
#    X_original, X_comb = feature_normalize(feature_column, price)
#    Y = np.array([[np.array(df['apply'])[i]] for i in range(len(df['apply']))])
#    lr_model = LogisticRegression(fit_intercept = False)
#    lr_model.fit(X_comb, Y)
#    #simulation_data_generation(lr_model, X_original, price)
#    print(lr_model.coef_)
    simulation_data_generation(lr_model, X_original, price)
    csv_to_gz()

#    for i in range(4):
#        df = pd.read_csv(VW_DS_DIR + 'ds_loan' + str(i + 1) +'_2.csv')
#        X = df[feature_column]
#        y = list(df['class'])
#        change_vw(VW_DS_DIR, 'ds_loan' + str(i + 1) +'_2', X, y)
    
