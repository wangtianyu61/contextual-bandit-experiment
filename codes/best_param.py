import numpy as np
import pandas as pd
import csv

VW_DS_DIR = 'vw_loan/'
#get the best param that achieves the minimum loss for each algorithm
def best_param(alg_class, alg_param_all, lr):
    
    return alg_param_all[alg_class[0]], lr[alg_class[0]]

#get the loss in each dataset for that param
def param_loss(alg_class, alg_param_all, lr):
    best_param_name, best_param_lr = best_param(alg_class, alg_param_all, lr)
    all_loss = pd.read_csv(VW_DS_DIR + 'loss_alg.csv')
    class_best = all_loss[all_loss['algo'] == best_param_name]
    class_best = class_best[class_best['lr'] == best_param_lr]
    print(class_best)
    return list(class_best['loss'])
    
if __name__ == '__main__':
    supervised = []
    epsilon_greedy_class = []
    greedy_class = []
    cover = []
    cover_nu = []
    bag = []
    bag_greedy = []
    avg_performance = pd.read_csv(VW_DS_DIR + 'avg_performance.csv')
    alg_param_all = list(avg_performance['algo'])
    param_split = [param.split(':') for param in alg_param_all]
    loss = list(avg_performance['loss'])
    lr = list(avg_performance['lr'])
    for index in range(len(loss)):
        loss_param = param_split[index]
        if loss_param[0] == 'epsilon':
            #print(loss_param)
            if loss_param[1] == str(0):
                greedy_class.append(index)
            else:
                epsilon_greedy_class.append(index)
        elif loss_param[0] == 'bag':
            if len(loss_param) == 3:
                bag.append(index)
            else:
                bag_greedy.append(index)
        elif loss_param[0] == 'cover':
            if len(loss_param) == 6:
                cover_nu.append(index)
            elif len(loss_param) == 3:
                cover.append(index)
            else:
                if loss_param[2] == 'psi':
                    cover.append(index)
                else:
                    cover_nu.append(index)
        elif loss_param[0] == 'supervised':
            supervised.append(index)
            
    csv_test = open(VW_DS_DIR + 'avg_test.csv', 'w', newline = '')
    csv_writer = csv.writer(csv_test)
    #csv_writer.writerow(['supervised'] + param_loss(supervised, alg_param_all, lr))
    csv_writer.writerow(['greedy'] + param_loss(greedy_class, alg_param_all, lr))
    csv_writer.writerow(['epsilon-greedy'] + param_loss(epsilon_greedy_class, alg_param_all, lr))
    csv_writer.writerow(['cover'] + param_loss(cover, alg_param_all, lr))
    csv_writer.writerow(['cover-nu'] + param_loss(cover_nu, alg_param_all, lr))
    csv_writer.writerow(['bag'] + param_loss(bag, alg_param_all, lr))
    csv_writer.writerow(['bag-greedy'] + param_loss(bag_greedy, alg_param_all, lr))
    csv_test.close()
    
    