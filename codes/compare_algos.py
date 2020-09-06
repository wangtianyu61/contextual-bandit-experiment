import pandas as pd
import numpy as np
from scipy.special import erf, erfinv

csvpath_ranking = 'vw_rank/avg_test.csv'
csvpath_ranking_1 = '../res/train_test_split/ms/test0.4.csv'

csvpath_openml = 'vw_openml/avg_test.csv'
csvpath_openml_1 = '../res/train_test_split/ms/test0.4.csv'

def significance(x, y, sz):
    diff = x - y
    se = 1e-6 + np.sqrt((x * (1 - x) + y * (1 - y))/ sz)
    #print(x, y, np.abs(diff / se))
    pval = 1 - erf(np.abs(diff / se))
    return pval


n_size_openml = [148, 625, 2000, 2000, 2000, 2000, 2000, 1728, 3196, 20000]
n_size_hotel = [1112, 429, 1306, 325, 278]
n_size_rank = [4000, 4000, 4000, 4000, 4000, 9978, 5064]



def compare_win_loss(csv_path, n_size, p_value):
    
    df = pd.read_csv(csv_path)
    df.columns = ['did'] + list(range(1, len(df.columns)))
    
    did_all = list(range(1, len(df.columns)))
    
    if 'supervised' in list(df['did']):
        df.drop([0], inplace = True)
        
    alg_all = list(df['did'])
    df_pvloss = np.array(df[did_all])
    win_loss = np.zeros((len(alg_all), len(alg_all)))
    for i in range(len(alg_all)):
        for j in range(i):
            for k in range(len(did_all)):
                sig_value = significance(df_pvloss[i][k], df_pvloss[j][k], n_size[k])
                #print("when ", alg_all[i], "compares", alg_all[j], " in did ", k, sig_value)
                if sig_value <= p_value:
                    if df_pvloss[i][k] < df_pvloss[j][k]:
                        win_loss[i][j] = win_loss[i][j] + 1
                        win_loss[j][i] = win_loss[j][i] - 1
                    elif df_pvloss[i][k] > df_pvloss[j][k]:
                        win_loss[i][j] = win_loss[i][j] - 1
                        win_loss[j][i] = win_loss[j][i] + 1

#    print(win_loss)
    return win_loss
                    
            
if __name__ == '__main__':
    
    win_loss_ranking = compare_win_loss(csvpath_ranking, n_size_rank, 0.05)
    win_loss_ranking_1 = compare_win_loss(csvpath_ranking_1, n_size_rank, 0.05)
    
    win_loss_openml = compare_win_loss(csvpath_openml, n_size_openml, 0.05)
    win_loss_openml_1 = compare_win_loss(csvpath_openml_1, n_size_openml, 0.05)
    print(win_loss_ranking)
    print(win_loss_ranking_1)
    print(win_loss_openml + win_loss_ranking)
    print(win_loss_openml_1 + win_loss_openml_1)
    
    