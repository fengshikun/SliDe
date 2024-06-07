# take pos normaliser parameters
import torch
import os
# borrow the pos norm para from target model
target_model = '/home/fengshikun/Pretraining-Denoising/experiments/ET-PCQM4MV2_dih_var0.04_var2_com_re_f2d_da/step=386103-epoch=7-val_loss=3.0421-test_loss=3.1239-train_per_step=3.1265.ckpt'


# source_model = '/home/fengshikun/Pretraining-Denoising/experiments/pretraining_f2d_pos_bat_torsion_w0.001_watchgradient2_fix_error_add_ring/step=399999-epoch=8-val_loss=0.7723-test_loss=0.5901-train_per_step=0.5615.ckpt'

source_model = '/home/fengshikun/Pretraining-Denoising/experiments/pretraining_f2d_pos_bat_torsion_w0.001_watchgradient2_fix_error/step=399999-epoch=8-val_loss=0.7284-test_loss=0.5286-train_per_step=0.5135.ckpt'


source_model = '/home/fengshikun/Pretraining-Denoising/experiments/pretraining_f2d_pos_bat_torsion_w0.001_watchgradient2_fix_error_add_ring_b0.04a0.05t1_0.2/step=399999-epoch=8-val_loss=0.7957-test_loss=0.7416-train_per_step=0.6559.ckpt'


source_model = '/home/fengshikun/BAT_Model/pretraining_f2d_pos_bat_torsion_w0.001_watchgradient2_fix_error_add_ring_b0.004a0.01t0.2_0.04_vnan/step=199999-epoch=14-val_loss=0.9612-test_loss=0.9103-train_per_step=0.9980.ckpt'


source_model = '/home/fengshikun/BAT_Model/pretraining_f2d_pos_bat_torsion_w0.001_watchgradient2_fix_error_add_ring_b0.04a0.01t0.2_0.04_vnan/step=199999-epoch=14-val_loss=0.7995-test_loss=0.7746-train_per_step=0.7539.ckpt'


source_model = '/home/fengshikun/BAT_Model/pretraining_f2d_pos_bat_torsion_w0.001_watchgradient2_fix_error_add_ring_b0.04a0.05t1_0.2_vnan/step=199999-epoch=14-val_loss=0.7844-test_loss=0.7572-train_per_step=0.7846.ckpt'

def load_model(filepath, source_model):
    ckpt = torch.load(filepath, map_location="cpu")
    ckpt2 = torch.load(source_model, map_location='cpu')
    args = ckpt["hyper_parameters"]



    state_dict = ckpt["state_dict"]
    
    sstate_dict = ckpt2['state_dict']
    
    # NOTE for debug
    pos_state_dict = {}
    for k, v in state_dict.items():
        if 'pos_normalizer' in k:
            # if 'output_model_noise.0' in k:
            #     k = k.replace('output_model_noise.0', 'output_model_noise')
            pos_state_dict[k] = v
            sstate_dict[k] = v
    
    torch.save(ckpt2, os.path.join(os.path.dirname(source_model), 'mix_pos.ckpt'))
    return

def load_model2(filepath):
    ckpt = torch.load(filepath, map_location="cpu")
    



    state_dict = ckpt["state_dict"]
    
    del state_dict['model.dmean']
    del state_dict['model.dstd']
    
    torch.save(ckpt, os.path.join(os.path.dirname(filepath), 'mix_pos.ckpt'))
    return
    
    

# load_model2('/data2/SKData/Pretraining-Denoising/logs/f2d_md17_sample128_frad_multask_Areg_omit_loss_nan_gcv1_benzene/step=233239-epoch=1959-val_loss=0.1266-test_loss=nan-train_per_step=0.0022.ckpt')
# load_model2('/data2/SKData/Pretraining-Denoising/logs/f2d_md17_sample128_frad_multask_Areg_omit_loss_nan_gcv1_benzene/step=223719-epoch=1879-val_loss=0.1266-test_loss=nan-train_per_step=0.0021.ckpt')


load_model2('/data2/SKData/Pretraining-Denoising/logs/f2d_md17_sample128_frad_multask_Areg_omit_loss_nan_gcv1_benzene/step=183259-epoch=1539-val_loss=0.1266-test_loss=nan-train_per_step=0.0020.ckpt')
# load_model(target_model, source_model)