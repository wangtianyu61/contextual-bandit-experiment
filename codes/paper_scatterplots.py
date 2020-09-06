import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from compare_algos import significance

plt.style.use('ggplot')

csvpath_ranking = 'vw_rank/avg_test.csv'
csvpath_ranking_1 = '../res/train_test_split/ms/test0.4.csv'

csvpath_openml = 'vw_openml/avg_test.csv'
csvpath_openml_1 = '../res/train_test_split/ms/test0.4.csv'

n_size_openml = [148, 625, 2000, 2000, 2000, 2000, 2000, 1728, 3196, 20000]
n_size_rank = [4000, 4000, 4000, 4000, 4000, 9978, 5064]


def scatterplot(raw_loss, alg_names, p_value, labels=None,
                lim_min=-0.25, lim_max=1.,fname=None):
    assert len(alg_names) == 2
    if labels is None:
        labels = alg_names

    sz = n_size_rank + n_size_openml
    
    pvals = np.zeros(len(sz))
    for i in range(len(sz)):
        pvals[i] = significance(raw_loss[1][i], raw_loss[0][i], sz[i])

    plt.figure(figsize=(6,6))
    # plt.scatter(x, y,
    #             s=plt.rcParams['lines.markersize']**2 * (pvals < args.alpha).map(lambda x: 0.7 if x else 0.2),
    #             c=(pvals < args.alpha).map(lambda x: 'r' if x else 'k'))
    
    sign_idxs = (pvals < p_value)
    nsign_idxs = np.logical_not(sign_idxs)
    plt.scatter(raw_loss[1][nsign_idxs], raw_loss[0][nsign_idxs], s=plt.rcParams['lines.markersize']**2 * 0.4, c='k')
    plt.scatter(raw_loss[1][sign_idxs], raw_loss[0][sign_idxs], s=plt.rcParams['lines.markersize']**2 * 0.7, c='r')
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], color='k')

    plt.xlabel(labels[1])
    plt.ylabel(labels[0])

    
    figname = alg_names[1] + 'vs' + alg_names[0]
    plt.title(figname)
    plt.savefig('fig_loss/'+ figname + '.png')
    plt.show()
    

def preprocessing(path1, path2):
    df1 = pd.read_csv(path1)
    df1.columns = ['did'] + list(range(1, len(df1.columns)))
    
    df2 = pd.read_csv(path2)
    df2.columns = ['did'] + list(range(1, len(df2.columns)))
    aggre_loss = np.zeros((7, len(df1.columns) + len(df2.columns) - 2))
    alg_list = ['epsilon-greedy', 'greedy', 'bag', 'bag-greedy', 'cover', 'cover-nu', 'FALCON']
    for index in range(len(alg_list)):
        aggre_loss[index] = np.array([float(i) for i in np.array(df1[df1['did'] == alg_list[index]])[0][1:]] + [float(i) for i in np.array(df2[df2['did'] == alg_list[index]])[0][1:]])
    return alg_list, aggre_loss

if __name__ == '__main__':
    i = 0
    alg_list, aggre_loss = preprocessing(csvpath_ranking, csvpath_openml)
    falcon_index = len(aggre_loss) - 1
    for i in range(len(aggre_loss) - 1):
        scatterplot(aggre_loss[[i, falcon_index]], [alg_list[i], alg_list[falcon_index]], 0.05)
    
