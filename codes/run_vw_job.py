import argparse
import os
import re
import subprocess
import sys
import time
import numpy as np
#from vowpalwabbit import pyvw

USE_ADF = True
USE_CS = False
RANDOM_TIE = False
data_choice = 'loan'
#where we store the vw.exe
VW = '../vw'

#the location of datasets and results
if data_choice == 'openml':
    VW_DS_DIR = '../datasets/vwdatasets/'
    DIR_PATTERN = '../res/cbresults_{}/'
elif data_choice == 'hotel':
    VW_DS_DIR = '../datasets/pricing/MSOM_Hotel/'
    DIR_PATTERN = '../res/cbresults_hotel_{}/'
elif data_choice == 'ms':
    VW_DS_DIR = '../datasets/MSLR-WEB10K/'
    DIR_PATTERN = '../res/cbresults_ms_{}/'
elif data_choice == 'yahoo':
    VW_DS_DIR = '../datasets/yahoo_dataset/'
    DIR_PATTERN = '../res/cbresults_yahoo_{}/'
elif data_choice == 'loan':
    VW_DS_DIR = '../datasets/OnlineAutoLoan/'
    DIR_PATTERN = '../res/cbresults_loan_{}/'
    
rgx = re.compile('^average loss = (.*)$', flags=re.M)


#all possible parameters
def expand_cover(policies):
    algs = []
    for psi in [0, 0.01, 0.1, 1.0]:
        algs.append(('cover', policies, 'psi', psi))
        algs.append(('cover', policies, 'psi', psi, 'nounif', None))
    return algs

params = {
    'alg': [
        ('supervised',),
        ('epsilon', 0),
        ('epsilon', 0.02),
        ('epsilon', 0.05),
        ('epsilon', 0.1),
        #-------------the package in 2018 does not have nounifagree, agree_mellowness.-------------------
#        ('epsilon', 0.02, 'nounifagree', None, 'agree_mellowness', 1e-2),
#        ('epsilon', 0.02, 'nounifagree', None, 'agree_mellowness', 1e-4),
#        ('epsilon', 0.02, 'nounifagree', None, 'agree_mellowness', 1e-6),
#        ('epsilon', 1, 'nounifagree', None, 'agree_mellowness', 1e-2),
#        ('epsilon', 1, 'nounifagree', None, 'agree_mellowness', 1e-4),
#        ('epsilon', 1, 'nounifagree', None, 'agree_mellowness', 1e-6),
        #==========================================
        ('bag', 2),
        ('bag', 4),
        ('bag', 8),
        ('bag', 16),
        
        ('bag', 2, 'greedify', None),
        ('bag', 4, 'greedify', None),
        ('bag', 8, 'greedify', None),
        ('bag', 16, 'greedify', None),
        #-------------the package in 2018 does not have regcb and regcbopt.-------------------
#        ('regcb', None, 'mellowness', 1e-1),
#        ('regcb', None, 'mellowness', 1e-2),
#        ('regcb', None, 'mellowness', 1e-3),
#        ('regcbopt', None, 'mellowness', 1e-1),
#        ('regcbopt', None, 'mellowness', 1e-2),
#        ('regcbopt', None, 'mellowness', 1e-3),
        #=============================================
        ('cover', 1),
        ('cover', 1, 'nounif', None),
        ] + expand_cover(4) + expand_cover(8) + expand_cover(16),
    'learning_rate': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    'cb_type': ['dr', 'ips', 'mtr'],
    }

extra_flags = None


#enumerate all possible choices in the parametric setting 51*9*3 = 1377
#in the left ones, there are still 39*9*3 = 1053 possibilities
def param_grid():
    grid = [{}]
    for k in params:
        new_grid = []
        for g in grid:
            for param in params[k]:
                gg = g.copy()
                gg[k] = param
                new_grid.append(gg)
        grid = new_grid

    return grid

#emuerate all possible datasets in the file path
def ds_files():
    import glob
    return sorted(glob.glob(os.path.join(VW_DS_DIR, '*.vw.gz')))


def get_task_name(ds, params):
    did, n_actions = os.path.basename(ds).split('.')[0].split('_')[1:]

    task_name = 'ds:{}|na:{}'.format(did, n_actions)
    if len(params) > 1:
        task_name += '|' + '|'.join('{}:{}'.format(k, v) for k, v in sorted(params.items()) if k != 'alg')
    task_name += '|' + ':'.join([str(p) for p in params['alg'] if p is not None])
    return task_name


def process(ds, params, results_dir, add_parse = None, data_choice = 'loan'):
    print('processing', ds, params)
    did, n_actions = os.path.basename(ds).split('.')[0].split('_')[1:3]

    cmd = [VW, ds, '-b', '24']
    
    for k, v in params.items():
        if k == 'alg':
            #supervised algorithms
            if v[0] == 'supervised':               
                if data_choice == 'ms' or data_choice == 'yahoo':
                    cmd += ['--csoaa_ldf']
                else:
                    cmd += ['--csoaa' if USE_CS else '--oaa', str(n_actions)]
            #bandit learning
            else:
                if data_choice == 'ms' or 'yahoo' or 'loan':
                    cmd += ['--cbify_ldf']
                else:
                    cmd += ['--cbify', str(n_actions)]
                if USE_CS:
                    cmd += ['--cbify_cs']
                if extra_flags:
                    cmd += extra_flags
                if USE_ADF:
                    cmd += ['--cb_explore_adf']
                if RANDOM_TIE:
                    cmd += ['--randomtie']
                assert len(v) % 2 == 0, 'params should be in pairs of (option, value)'
                for i in range(int(len(v) / 2)):
                    print(v[2*i], v[2*i + 1])
                    cmd += ['--{}'.format(v[2 * i])]
                    if v[2 * i + 1] is not None:
                        cmd += [str(v[2 * i + 1])]
        else:
            if params['alg'][0] == 'supervised' and k == 'cb_type':
                pass
            else:
                cmd += ['--{}'.format(k)] + [str(v)]
    if add_parse == True:
        cmd += ['--progress'] + [str(1)]
        
    print('running', cmd)
   
    t = time.time()
    #for the subprocess to finish
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    sys.stderr.write('\n\n{}, {}, time: {}, output:\n'.format(ds, params, time.time() - t))
    output = str(output, encoding = 'utf-8')
    sys.stderr.write(output)
    pv_loss = float(rgx.findall(output)[0])
    print ('elapsed time:', time.time() - t, 'pv loss:', pv_loss)

    return pv_loss, output


def skip_params(params):
    # skip evaluating the following
    return (params['alg'][0] == 'supervised' and params['cb_type'] != 'mtr') or \
            (params['alg'][0].startswith('regcb') and params['cb_type'] != 'mtr')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vw job')
    parser.add_argument('task_id', type=int, help='task ID, between 0 and num_tasks - 1')
    parser.add_argument('num_tasks', type=int)
    parser.add_argument('--task_offset', type=int, default=0,
                        help='offset for task_id in output filenames')
    
    #the results table name
    parser.add_argument('--results_dir', default=DIR_PATTERN.format('agree01'))
    parser.add_argument('--name', default=None)
    
    #if add the argument --test, the last steps will not be implemented 
    parser.add_argument('--test', action='store_true')
    
    parser.add_argument('--flags', default=None, help='extra flags for cb algorithms')
    
    args = parser.parse_args()

    if args.name is not None:
        args.results_dir = DIR_PATTERN.format(args.name)

    if args.flags is not None:
        extra_flags = args.flags.split()

    #all tasks taht needed to run
    grid = param_grid()
    dss = ds_files()
    tot_jobs = len(grid) * len(dss)

    if args.task_id == 0:
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
            import stat
            #change to the right of write
            os.chmod(args.results_dir, os.stat(args.results_dir).st_mode | stat.S_IWOTH)
    else:
        while not os.path.exists(args.results_dir):
            print('dead loop?')
            time.sleep(1)
    
    if not args.test:
        print("yes")
        fname = os.path.join(args.results_dir, 'all_losses.txt')
        done_tasks = set()
        print(fname)
        if os.path.exists(fname):
            #remains a question whether to add [0] after line.split()
            #if there is no empty line after, we should add[0]
            done_tasks = set([line.split()[0] for line in open(fname).readlines()])
        #stop to the finish line.
        loss_file = open(fname, 'a')
        #loss_file.write("\n")
    idx = args.task_id
    count = 0
    loss = []
    while idx < tot_jobs:
        print(idx)
        #for the dataset to use.
        ds = dss[int(idx / len(grid))]
        #for the method to use
        params = grid[idx % len(grid)]
        if args.test:
            print(ds, params)
        else:
            task_name = get_task_name(ds, params)
            if task_name not in done_tasks and not skip_params(params):
                try:
                    pv_loss, all_output = process(ds, params, args.results_dir, None, 'loan')
                    loss.append(pv_loss)
                    loss_file.write('{} {}\n'.format(task_name, pv_loss))
                    loss_file.flush()
                    os.fsync(loss_file.fileno())
                except subprocess.CalledProcessError:
                    sys.stderr.write('\nERROR: TASK FAILED {} {}\n\n'.format(ds, params))
                    print('ERROR: TASK FAILED', ds, params)
        idx += args.num_tasks
    print("average loss", np.mean(loss), " maximum loss", np.max(loss), " minimum loss", np.min(loss))
    if not args.test:
        loss_file.close()
