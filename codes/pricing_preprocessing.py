import pandas as pd
import numpy as np
import os, gzip
import scipy.sparse as sp

Hotel_DIR = '../datasets/pricing/MSOM_Hotel/'
Hotel_ID = 5



def map_class(y):
    ## map y
    arm = list(set(list(y)))
    arm_number = len(arm)
    #use the string for further locating values
    arm_label = []
    arm_map = dict(zip(arm, list(range(0, arm_number))))
    for item in list(y):
        arm_label.append(arm_map[item])

    return arm_label

#shuffle the value of x and y to a stochastic order
def shuffle(X, y):
    n = X.shape[0]
    perm = np.random.permutation(n)
    #print(n, perm)
    X_shuf = X.loc[perm]
    y_shuf = y[perm]
    return X_shuf, y_shuf

def feature_engineering(Hotel_ID):
    df = pd.read_csv(Hotel_DIR + 'Hotel' + str(Hotel_ID) +'.csv')
    feature_column = ['Product_ID','Distribution_Channel','Advance_Purchase','Party_Size','Length_of_Stay','Number_of_Rooms','Nightly_Rate',
                  'Membership_Status','VIP_Membership_Status']

    #delete the dataset with WAIT and other possibilities
    #i.e. fewer choices
    df = df.drop(df[df['Purchased_Room_Type'] == 'WAIT'].index)
    Room_type = list(set(list(df['Purchased_Room_Type'])))
    df = df[df['Product_ID'] > len(Room_type)]

    #delete the repetitive row with the same Booking ID
    df.drop_duplicates(subset = 'Booking_ID', keep = 'last', inplace = True)
    
    #discretize encoding
    df_select_y = pd.Series(map_class(df['Purchased_Room_Type']))
    df['Distribution_Channel'] = map_class(df['Distribution_Channel'])
    df_select_X = df[feature_column]
    df_select_X.index = np.array(range(len(df_select_X)))
    X, y = shuffle(df_select_X, df_select_y)
    X['class'] = y
    
    n_classes = max(y) + 1 
    print("check step")
    fname = 'hotel_{}_{}'.format(Hotel_ID, n_classes)
    #add it to the csv path for the use of bypassing the monster
    
    X.to_csv(Hotel_DIR + fname + '.csv', index = False)

    with gzip.open(os.path.join(Hotel_DIR, fname + '.vw.gz'), 'w') as f:
        #if the matrix is sparse then we use the nonzero value to fill.
        if sp.isspmatrix_csr(X):
            for i in range(X.shape[0]):
                str_output = '{} | {}\n'.format(y[i] + 1, ' '.join(
                    '{}:{:.6f}'.format(j, val) for j, val in zip(list(X.loc[i].index), list(X.loc[i]))))
                f.write(str_output.encode())
        else:
            for i in range(X.shape[0]):
                str_output = '{} | {}\n'.format(y[i] + 1, ' '.join(
                    '{}:{:.6f}'.format(j, val) for j, val in enumerate(X.loc[i]) if val != 0))
                f.write(str_output.encode())



if __name__ == '__main__':
    for index in range(1,6):
        feature_engineering(index)