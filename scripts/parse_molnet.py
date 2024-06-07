import re
import numpy as np
import pandas as pd
import argparse
import os

def parse_log(file_path, regression=False, mae=False):
    
    with open(file_path, 'r') as file:
        log = file.read()
    
    if regression:
        if mae:
            pattern_valid = r"valid mae is (\d+\.\d+)"
            pattern_test = r"test mae is (\d+\.\d+)"
        else: 
            pattern_valid = r"val rmse is (\d+\.\d+)"
            pattern_test = r"test rmse is (\d+\.\d+)"
        
    else:
        pattern_valid = r"valid auc is (\d+\.\d+)"
        pattern_test = r"test auc is (\d+\.\d+)"

    matches_valid = re.findall(pattern_valid, log)
    matches_test = re.findall(pattern_test, log)

    # print("Valid Results:")
    epoch_res = []
    for match in matches_valid:
        valid_res = match
    # print("\nTest Results:")
    for match in matches_test:
        test_res = match

    epoch_res.append([float(valid_res), float(test_res)])
    
    # TODO, return the test_res coorespond to the best valid_res(regression: lower is better, otherwise higher is better)
    if regression:
        test_res = min(epoch_res, key=lambda x: x[0])[1]
    else:
        test_res = max(epoch_res, key=lambda x: x[0])[1]
    return test_res


# print(parse_log(file_path, regression=True))


tasks = ['bbbp', 'bace', 'clintox', 'tox21', 'toxcast', 'sider', 'hiv', 'pcba', 'muv', 'esol', 'freesolv', 'lipo', 'qm7dft', 'qm8dft', 'qm9dft']


tasks = ['bbbp', 'bace', 'clintox', 'tox21', 'toxcast', 'hiv', 'muv', 'esol', 'lipo']

# tasks = ['qm7dft', 'qm8dft', 'qm9dft']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parsing log files with specific prefix.')
    parser.add_argument('--log_prefix', type=str, default='molnet_unimol_data_full_f2d_frad', help='The prefix of log files.')
    # parser.add_argument('--log_prefix', type=str, default='unimol/pretrain_unimol_kp300w', help='The prefix of log files.')
    args = parser.parse_args()

    log_prefix = args.log_prefix


    task_rst_lst = []
    for task in tasks:
        task_res = []
        for seed in [1, 2, 3]:
            log_file = f'{log_prefix}_{task}_seed{seed}.log'
            if not os.path.exists(log_file):
                break
            mae = False
            if task in ['esol', 'freesolv', 'lipo', 'qm7dft', 'qm8dft', 'qm9dft']:
                regression = True
                if task in ['qm7dft', 'qm8dft', 'qm9dft']:
                    mae = True
            else:
                regression = False
            try:
                task_res.append(parse_log(log_file, regression, mae))
            except Exception as e:
                print(f'error happens parsing the result of task {task}, seed{seed}')
                break
        
        task_mean_std = f'{np.mean(task_res):.4f}({np.std(task_res):.4f})'
        task_rst_lst.append([task, task_mean_std])

    with open('molnet_res.txt', 'w') as mw:
        mw.write('\t'.join([ele[1] for ele in task_rst_lst]))
    df = pd.DataFrame(task_rst_lst, columns=['Task', 'Mean (Standard Deviation)'])
    print(df)