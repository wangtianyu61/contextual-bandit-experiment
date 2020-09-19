import pandas as pd
import numpy as np


VW_DS_DIR = '../../datasets/OnlineAutoLoan/test'
#ds_loan_10_1000.csv represents the case where we have 6 true. Suppose we use 3 at first.

csv_path = '/ds_loan_10_1000.csv'

arm_number = int(csv_path.split('_')[-2])

df = pd.read_csv(VW_DS_DIR + csv_path)
total_round = len(df)/arm_number
context_index = df.columns[2:]

for t in range(total_round):
    #select the context info in each round
    context_info = np.array(df[context_index].loc[arm_number*t:arm_number*(t + 1)])
    #select the reward info in each round
    reward = list(df['reward'][arm_number*t:arm_number*(t + 1)])
    arm_index_1 = choose_arm(context_info, theta_hat_1, all_previous_context_1, t)[0]
    
    
    arm_index_2 = choose_arm(context_info, theta_hat_2, all_previous_context_2, t)[0]
    alpha1 = noise_std*math.sqrt(context_sub*math.log(1 + 2*t/context_sub)/2 + math.log(1/delta))
    residual[t - exploration_period] = alpha1*math.sqrt(np.dot(context_info[arm_index_2][0:context_sub], np.linalg.inv(all_previous_context_1)).dot(context_info[arm_index_2][0:context_sub]))
    
    arm_index_3 = choose_arm(context_info, theta_hat_3, all_previous_context_3, t)[0]
    context_history[t - exploration_period] = context_info[arm_index_2]
    theta_hat_history[t - exploration_period] = theta_hat_2
    
    #check the condition of whether to switch
    if t > exploration_period and model_status == 'simple':
        diff[t - exploration_period + 1], model_status = checking_condition(context_history, theta_hat_history, receive_reward_1, residual, t)
        #if model
        #print(np.dot(context_history[t - exploration_period - 1], theta_hat_history[t - exploration_period -1]), receive_reward_1[t - 1])
    best_index = best_arm(theta, context_info, choice)
    
    
    simple_model[t + 1] = reward[arm_index_1] + simple_model[t]
    
    true_complex_model[t + 1] = reward[arm_index_3] + true_complex_model[t]
    best_model[t + 1] = reward[best_index] + best_model[t]
    
    use_context_1.append(context_info[arm_index_1][0: context_sub])
    if model_status == 'complex':
        use_context_2.append(context_info[arm_index_2])
        receive_reward_2.append(reward[arm_index_2])
        
        complex_model[t + 1] = reward[arm_index_2] + complex_model[t]
    else:
        use_context_2.append(context_info[arm_index_1])
        receive_reward_2.append(reward[arm_index_1])
        complex_model[t + 1] = reward[arm_index_1] + complex_model[t]
        
    use_context_3.append(context_info[arm_index_3])
    
    receive_reward_1.append(reward[arm_index_1])
    
    receive_reward_3.append(reward[arm_index_3])
#        print(np.array(context_info[arm_index_1][0: context_sub]).shape)
    all_previous_context_1 += np.dot(np.array([context_info[arm_index_1][0: context_sub]]).T, np.array([context_info[arm_index_1][0:context_sub]]))
    all_previous_context_2 += np.dot(np.array([context_info[arm_index_1]]).T, np.array([context_info[arm_index_1]]))
    all_previous_context_3 += np.dot(np.array([context_info[arm_index_3]]).T, np.array([context_info[arm_index_3]]))
   
    #diff[t - exploration_period + 1] = diff[t - exploration_period] + np.dot(theta_hat_2, context_info[arm_index_2][0]) - np.dot(theta_hat_1, context_info[arm_index_2][0][0: context_sub])
        
        
    regret_figure([best_model - simple_model, best_model - complex_model, best_model - true_complex_model], choice)