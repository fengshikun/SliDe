# python scripts/test_molnet.py --log_prefix f2d_frad --cu_list 0 1 2 3 --pretrained_model /data/protein/SKData/DenoisingData/Model_bak/test_model/pretraining_f2d_pos_frad/step=211151-epoch=7-val_loss=0.1345-test_loss=0.1439-train_per_step=0.1357.ckpt
# python scripts/test_molnet.py --log_prefix f2d_mask0.15_frad --cu_list 0 1 3 --pretrained_model /data/protein/SKData/DenoisingData/pretraining_f2d_pos_frad_debug_ew0.1_pna_mask0.15/step=249999-epoch=8-val_loss=6678168.0000-test_loss=384682.5938-train_per_step=7.7368.ckpt
# python scripts/test_molnet.py --log_prefix f2d_mask0.2_frag_gf --cu_list 1 2 3 --pretrained_model /data/protein/SKData/Pretraining-Denoising_exf/tmp/step=249999-epoch=8-val_loss=0.2624-test_loss=0.2542-train_per_step=0.2131.ckpt


# python scripts/test_molnet.py --log_prefix pretraining_f2d_pos_frad_ctl_gf_frag_mask0.2_unipcq_ep2 --cu_list 1 2 3 --pretrained_model /data/protein/SKData/DenoisingData/DeCL/pretraining_f2d_pos_frad_ctl_gf_frag_mask0.2_unipcq/step=225289-epoch=2-val_loss=0.3073-test_loss=0.2361-train_per_step=0.2483.ckpt

# python scripts/test_molnet.py --log_prefix pretraining_f2d_pos_frad_debug_ew0.1_pna_mask0.15_lh --cu_list 1 2 3 --pretrained_model /data/protein/SKData/DenoisingData/DeCL/pretraining_f2d_pos_frad_debug_ew0.1_pna_mask0.15/step=249999-epoch=8-val_loss=6678168.0000-test_loss=384682.5938-train_per_step=7.7368.ckpt 

import sys
sys.path.append(sys.path[0]+'/..')
from torchmdnet.datasets.molnet import MolNetDataset
import argparse
parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--log_prefix', type=str, default='wo_pretrain', help='Log prefix argument')
parser.add_argument('--cu_list', nargs='+', type=int, default=[5, 6], help='List of cu arguments')
parser.add_argument('--pretrained_model', type=str, default='', help='Pretrained model argument')
parser.add_argument('--output_model', type=str, default='ScalarMol2', help='Pretrained model argument') # default linear regression

args = parser.parse_args()

log_prefix = args.log_prefix
cu_list = args.cu_list
pretrained_model = args.pretrained_model

task_cmd_lst = []
exe_cmd_lst = []
for cid in cu_list:
    task_cmd_lst.append(open(f'{cid}_mol_cmd.sh', 'w'))
    exe_cmd_lst.append(f'sh {cid}_mol_cmd.sh > {cid}_mol_cmd.log 2>&1 &')


# all_tasks = ['bbbp', 'bace', 'clintox', 'tox21', 'toxcast', 'sider', 'hiv', 'muv', 'esol', 'freesolv', 'lipo']

all_tasks = ['bbbp', 'bace', 'clintox', 'tox21', 'toxcast', 'sider', 'hiv', 'muv', 'esol', 'freesolv', 'lipo']

all_tasks = ['clintox', 'sider']
reg_tasks = ['esol', 'freesolv', 'lipo']


all_tasks = ['bbbp', 'bace', 'clintox', 'tox21', 'toxcast', 'hiv', 'muv', 'esol', 'lipo']


all_tasks = ['bbbp', 'bace', 'clintox', 'tox21', 'toxcast', 'esol', 'lipo']

all_tasks = ['hiv', 'muv']
# all_tasks = ['bbbp']

base_dir = '/data/protein/SKData/DenoisingData/molecular_property_prediction/'





# todo add seed
cu_idx = 0
device_num = len(cu_list)
for task in all_tasks:


    if task in ['hiv', 'muv']:
        train_epochs = 20
    else:
        train_epochs = 100
    batch_size = 64

    mol_dataset = MolNetDataset(f'{base_dir}/{task}/train.lmdb', dataset_name=task)
    train_size = len(mol_dataset)
    print(f'task {task}, train sample number: {train_size}')
    step_num = (train_size * train_epochs) // batch_size

    if task in reg_tasks:
        cmd_suffix = '--standardize true '
    else:
        cmd_suffix = ' '

    if pretrained_model:
        cmd_suffix += f'--pretrained-model {pretrained_model}'
    
    if task == 'clintox':
        output_model = 'ScalarMol'
    else:
        output_model = args.output_model
    
    
    for seed in [0, 1, 2]:
        cu_idx = cu_idx % device_num
        device_id = cu_list[cu_idx]
        
        test_cmd = f'CUDA_VISIBLE_DEVICES={device_id} python -u scripts/train.py --conf examples/ET-Molnet.yaml --layernorm-on-vec whitened --job-id {log_prefix}_{task}_{seed} --dataset-arg {task} --bond-length-scale 0 --denoising-weight 0.1 --test-interval 1 --lr-cosine-length {step_num} --num-steps {step_num}  --seed {seed}  {cmd_suffix} --output-model {args.output_model} > molnet_{log_prefix}_{task}_seed{seed}.log 2>&1'
        
        task_cmd_lst[cu_idx].write(f'{test_cmd}\n') 
        print(test_cmd)
        cu_idx += 1



for hdl in task_cmd_lst:
    hdl.close()

# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/fengshikun/miniconda3/envs/MOLuni/lib:$LD_LIBRARY_PATH
print('\n'.join(exe_cmd_lst))