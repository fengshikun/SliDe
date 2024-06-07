import numpy as np  # sometimes needed to avoid mkl-service error
import sys
sys.path.append(sys.path[0]+'/..')
import os
import argparse
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from torchmdnet.module import LNNP
from torchmdnet import datasets, priors, models
from torchmdnet.data import DataModule
from torchmdnet.models import output_modules
from torchmdnet.models.utils import rbf_class_mapping, act_class_mapping
from torchmdnet.utils import LoadFromFile, LoadFromCheckpoint, save_argparse, number
from pathlib import Path
import wandb

from torchmdnet.anix_datamodule import ANIXDataModule
from torchmdnet.iso17_datamodule import ISO17DataModule
from torchmdnet.md22_datamodule import MD22DataModule


wandb.login(key='a46eaf1ea4fdcf3a2a93022568aa1c730c208b50')
def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--load-model', action=LoadFromCheckpoint, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    parser.add_argument('--num-epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--num-steps', default=-1, type=int, help='Maximum number of gradient steps.')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--inference-batch-size', default=None, type=int, help='Batchsize for validation and tests.')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr-schedule', default="reduce_on_plateau", type=str, choices=['cosine', 'reduce_on_plateau', 'cosine_warmup'], help='Learning rate schedule.')
    parser.add_argument('--lr-patience', type=int, default=10, help='Patience for lr-schedule. Patience per eval-interval of validation')
    parser.add_argument('--lr-min', type=float, default=1e-6, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-factor', type=float, default=0.8, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-warmup-steps', type=int, default=0, help='How many steps to warm-up over. Defaults to 0 for no warm-up')
    parser.add_argument('--lr-cosine-length', type=int, default=400000, help='Cosine length if lr_schedule is cosine.')
    parser.add_argument('--early-stopping-patience', type=int, default=30, help='Stop training after this many epochs without improvement')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay strength')
    parser.add_argument('--ema-alpha-y', type=float, default=1.0, help='The amount of influence of new losses on the exponential moving average of y')
    parser.add_argument('--ema-alpha-dy', type=float, default=1.0, help='The amount of influence of new losses on the exponential moving average of dy')
    parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--log-dir', '-l', default='/tmp/logs', help='log file')
    parser.add_argument('--splits', default=None, help='Npz with splits idx_train, idx_val, idx_test')
    parser.add_argument('--train-size', type=number, default=None, help='Percentage/number of samples in training set (None to use all remaining samples)')
    parser.add_argument('--val-size', type=number, default=0.05, help='Percentage/number of samples in validation set (None to use all remaining samples)')
    parser.add_argument('--test-size', type=number, default=0.1, help='Percentage/number of samples in test set (None to use all remaining samples)')
    parser.add_argument('--test-interval', type=int, default=10, help='Test interval, one test per n epochs (default: 10)')
    parser.add_argument('--save-interval', type=int, default=10, help='Save interval, one save per n epochs (default: 10)')
    parser.add_argument('--save-top-k', type=int, default=10, help='save the topk model, -1 save all')

    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data prefetch')
    parser.add_argument('--redirect', type=bool, default=False, help='Redirect stdout and stderr to log_dir/log')
    parser.add_argument('--wandb-notes', default="", type=str, help='Notes passed to wandb experiment.')
    parser.add_argument('--job-id', default="auto", type=str, help='Job ID. If auto, pick the next available numeric job id.')
    parser.add_argument('--pretrained-model', default=None, type=str, help='Pre-trained weights checkpoint.')

    # dataset specific
    parser.add_argument('--dataset', default=None, type=str, choices=datasets.__all__, help='Name of the torch_geometric dataset')
    parser.add_argument('--dataset-root', default='data', type=str, help='Data storage directory (not used if dataset is "CG")')

    parser.add_argument('--dataset-idx', default='data', type=str, help='data idx for pocket dataset')

    parser.add_argument('--dataset-arg', default=None, type=str, help='Additional dataset argument, e.g. target property for QM9 or molecule for MD17')
    parser.add_argument('--coord-files', default=None, type=str, help='Custom coordinate files glob')
    parser.add_argument('--embed-files', default=None, type=str, help='Custom embedding files glob')
    parser.add_argument('--energy-files', default=None, type=str, help='Custom energy files glob')
    parser.add_argument('--force-files', default=None, type=str, help='Custom force files glob')
    parser.add_argument('--energy-weight', default=1.0, type=float, help='Weighting factor for energies in the loss function')
    parser.add_argument('--force-weight', default=1.0, type=float, help='Weighting factor for forces in the loss function')
    parser.add_argument('--position-noise-scale', default=0., type=float, help='Scale of Gaussian noise added to positions.')
    parser.add_argument('--denoising-weight', default=0., type=float, help='Weighting factor for denoising in the loss function.')
    parser.add_argument('--denoising-only', type=bool, default=False, help='If the task is denoising only (then val/test datasets also contain noise).')

    # bais specific
    # parser.add_argument('--sample-number', default=0., type=float, help='Bias sample number.')
    parser.add_argument('--sdf-path', default=None, type=str, help='sdf path')
    parser.add_argument('--violate', type=bool, default=False, help='violate conformation rules or not.')

    parser.add_argument('--md17', type=bool, default=False, help='is md17 test set.')
    parser.add_argument('--seperate-noise', type=bool, default=False, help='seperate noise.')

# bond_var=0.04, angle_var=0.143, torsion_var_r=2.8, torsion_var=0.41, coord_var=0.04

    parser.add_argument('--dihedral-angle-noise-scale', default=2.8, type=float, help='dihedral angle noise.')

    parser.add_argument('--angle-noise-scale', default=0.143, type=float, help='angle noise scale.')
    parser.add_argument('--bond-length-scale', default=0.04, type=float, help='bond length scale.')
    parser.add_argument('--nt-noise-scale', default=0.41, type=float, help='non rotate torsion length scale.')
    parser.add_argument('--use-edge-feat', default=False, type=bool, help='use edge feature')

    parser.add_argument('--add-ring-noise', default=False, type=bool, help='add ring noise')



    parser.add_argument('--composition', type=bool, default=False, help='violate conformation rules or not.')

    parser.add_argument('--equilibrium', type=bool, default=False, help='equilibrium.')
    parser.add_argument('--eq_weight', type=bool, default=False, help='eq_weight.')
    parser.add_argument('--cod_denoise', type=bool, default=False, help='cod_denoise.')
    parser.add_argument('--rdkit_conf', type=bool, default=False, help='rdkit_conf.')
    parser.add_argument('--integrate_coord', type=bool, default=False, help='integrate_coord.')

    parser.add_argument('--reverse_half', type=bool, default=False, help='reverse_half.')
    parser.add_argument('--addh', type=bool, default=False, help='add H atom.')

    parser.add_argument('--decay', type=bool, default=False, help='violate conformation rules or not.')
    parser.add_argument('--decay_coe', type=float, default=0.2, help='violate conformation rules or not.')

    parser.add_argument('--force_field', type=bool, default=False, help='predict force field')
    parser.add_argument('--pred_noise', type=bool, default=False, help='predict the noise')

    parser.add_argument('--no-target-mean', type=bool, default=False, help='violate conformation rules or not.')
    parser.add_argument('--sep-noisy-node', type=bool, default=False, help='sep noisy node')
    #sep-noisy-node
    parser.add_argument('--train-loss-type', type=str, default='', help='train loss type')


    parser.add_argument('--mask_atom', type=bool, default=False, help='mask atom or not')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='mask ratio')

    # model architecture
    parser.add_argument('--model', type=str, default='graph-network', choices=models.__all__, help='Which model to train')
    parser.add_argument('--output-model', type=str, default='Scalar', choices=output_modules.__all__, help='The type of output model')
    # output_model_ff_noise
    parser.add_argument('--output-model-ff-noise', type=str, default='Scalar', choices=output_modules.__all__, help='The type of output model')
    parser.add_argument('--prior-model', type=str, default=None, choices=priors.__all__, help='Which prior model to use')
    parser.add_argument('--output-model-noise', type=str, default=None, choices=output_modules.__all__ + ['VectorOutput'], help='The type of output model for denoising')

    # architectural args
    parser.add_argument('--embedding-dimension', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of interaction layers in the model')
    parser.add_argument('--num-rbf', type=int, default=64, help='Number of radial basis functions in model')
    parser.add_argument('--activation', type=str, default='silu', choices=list(act_class_mapping.keys()), help='Activation function')
    parser.add_argument('--rbf-type', type=str, default='expnorm', choices=list(rbf_class_mapping.keys()), help='Type of distance expansion')
    parser.add_argument('--trainable-rbf', type=bool, default=False, help='If distance expansion functions should be trainable')
    parser.add_argument('--neighbor-embedding', type=bool, default=False, help='If a neighbor embedding should be applied before interactions')
    parser.add_argument('--aggr', type=str, default='add', help='Aggregation operation for CFConv filter output. Must be one of \'add\', \'mean\', or \'max\'')

    # Transformer specific
    parser.add_argument('--distance-influence', type=str, default='both', choices=['keys', 'values', 'both', 'none'], help='Where distance information is included inside the attention')
    parser.add_argument('--attn-activation', default='silu', choices=list(act_class_mapping.keys()), help='Attention activation function')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--layernorm-on-vec', type=str, default=None, choices=['whitened'], help='Whether to apply an equivariant layer norm to vec features. Off by default.')

    # other args
    parser.add_argument('--derivative', default=False, type=bool, help='If true, take the derivative of the prediction w.r.t coordinates')
    parser.add_argument('--cutoff-lower', type=float, default=0.0, help='Lower cutoff in model')
    parser.add_argument('--cutoff-upper', type=float, default=5.0, help='Upper cutoff in model')
    parser.add_argument('--atom-filter', type=int, default=-1, help='Only sum over atoms with Z > atom_filter')
    parser.add_argument('--max-z', type=int, default=100, help='Maximum atomic number that fits in the embedding matrix')
    parser.add_argument('--max-num-neighbors', type=int, default=32, help='Maximum number of neighbors to consider in the network')

    parser.add_argument('--pocket-mnum', type=int, default=800, help='Maximum number of pocket')
    parser.add_argument('--lp-sep', type=bool, default=False, help='If true, multiply prediction by dataset std and add mean')

    parser.add_argument('--bat-noise', type=bool, default=False, help='if add bat noise in PCQM4MV2_Dihedral2')


    parser.add_argument('--standardize', type=bool, default=False, help='If true, multiply prediction by dataset std and add mean')
    parser.add_argument('--reduce-op', type=str, default='add', choices=['add', 'mean'], help='Reduce operation to apply to atomic predictions')

    parser.add_argument('--derivative-var', type=float, default=0.0, help='')
    parser.add_argument('--sample-num', type=int, default=32, help='')
    # fmt: on

    # for gradient clipping
    parser.add_argument('--gradient-clip-val', type=float, default=0.0, help='')

    # for noise type
    # for class PCQM4MV2_Force
    # add_noise_type: 0, BAT noise
    # add_noise_type: 1, Frad noise
    # add_noise_type: 2, coordinate guassian noise
    parser.add_argument('--add-noise-type', type=int, default=0, help='')

    # use force param or not
    parser.add_argument('--use-force-param', type=int, default=0, help='')

    parser.add_argument('--frad-guassian-scale', type=float, default=0.0, help='') # 0.04 to add frad as multi-task


    parser.add_argument('--random-sample', type=int, default=-1, help='') # random sample from pre-train data to verify the effect of scale of pre-training data

    parser.add_argument('--num-spherical', type=int, default=3, help='')

    # used in oc20 finetune
    parser.add_argument('--oc20-train-lmdb', type=str, help='oc20 train lmdb')
    parser.add_argument('--oc20-val-lmdb', type=str, help='oc20 val lmdb')
    parser.add_argument('--oc20-test-lmdb', type=str, help='oc20 test lmdb')
    parser.add_argument('--warm-epoch', default=1, type=int, help='warm up epoch')
    parser.add_argument('--val-check-interval', type=float, default=1, help='')
    
    
    # used for 3d cl K and T
    parser.add_argument('--K', type=int, default=65536, help='')

    # use force param or not
    parser.add_argument('--T', type=float, default=0.2, help='')
    
    parser.add_argument('--drop-last-batch', type=int, default=0, help='')
    
    


    args = parser.parse_args()

    if args.job_id == "auto":
        assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1, "Might be problematic with DDP."
        if Path(args.log_dir).exists() and len(os.listdir(args.log_dir)) > 0:
            next_job_id = str(max([int(x.name) for x in Path(args.log_dir).iterdir() if x.name.isnumeric()])+1)
        else:
            next_job_id = "1"
        args.job_id = next_job_id

    args.log_dir = str(Path(args.log_dir, args.job_id))
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if args.redirect:
        sys.stdout = open(os.path.join(args.log_dir, "log"), "w")
        sys.stderr = sys.stdout
        logging.getLogger("pytorch_lightning").addHandler(
            logging.StreamHandler(sys.stdout)
        )

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size

    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    return args


def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)
    # print(args)



    if "ANI1X" in args.dataset:
        data = ANIXDataModule(args)
    elif "ISO17" in args.dataset:
        data = ISO17DataModule(args)
    elif "MD22" in args.dataset:
        data = MD22DataModule(args)
    else:
    # initialize data module
        data = DataModule(args)
    data.prepare_data()
    data.setup("fit")
    
    args.lr_warmup_steps = round(len(data.train_dataset) * args.warm_epoch / args.batch_size)
    args.num_steps = round(len(data.train_dataset) * args.num_epochs / args.batch_size) 
    args.lr_cosine_length = round(len(data.train_dataset) * args.num_epochs / args.batch_size) 
    print(args)

    prior = None
    if args.prior_model:
        assert hasattr(priors, args.prior_model), (
            f"Unknown prior model {args['prior_model']}. "
            f"Available models are {', '.join(priors.__all__)}"
        )
        # initialize the prior model
        prior = getattr(priors, args.prior_model)(dataset=data.dataset)
        args.prior_args = prior.get_init_args()

    # initialize lightning module
    if 'MolNetDataset' in args.dataset:
        is_cls = data.train_dataset.classification
        setattr(args, 'is_cls', is_cls)
        if is_cls: # task number
            cls_number = data.train_dataset.cls_number
            setattr(args, 'cls_number', cls_number)


    model = LNNP(args, prior_model=prior, mean=data.mean, std=data.std)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        monitor="val_loss",
        # save_top_k=args.save_top_k,  # -1 to save all
        save_top_k=0,  # -1 to save all
        every_n_epochs=args.save_interval,
        # period=args.save_interval,
        filename="{step}-{epoch}-{val_loss:.4f}-{test_loss:.4f}-{train_per_step:.4f}",
        save_last=False,
    )
    
    if 'MolNetDataset' in args.dataset:
        early_stopping = EarlyStopping("val_roc_auc", patience=args.early_stopping_patience, mode='max')
    else:
        early_stopping = EarlyStopping("val_loss", patience=args.early_stopping_patience)

    tb_logger = pl.loggers.TensorBoardLogger(
        args.log_dir, name="tensorbord", version="", default_hp_metric=False
    )
    csv_logger = CSVLogger(args.log_dir, name="", version="")
    wandb_logger = WandbLogger(name=args.job_id, project='pre-training-via-denoising2', notes=args.wandb_notes, settings=wandb.Settings(start_method='fork', code_dir="."))

    # wandb_logger.watch(model) # log gradient
    @rank_zero_only
    def log_code():
        wandb_logger.experiment # runs wandb.init, so then code can be logged next
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))

    log_code()

    ddp_plugin = None
    if "ddp" in args.distributed_backend:
        # ddp_plugin = DDPStrategy(find_unused_parameters=True) # , num_nodes=args.num_nodes)
        ddp_plugin = DDPStrategy(find_unused_parameters=False) # , num_nodes=args.num_nodes)
        
        
    if args.val_check_interval == 0.:
        val_check_interval = None
    else:
        val_check_interval = args.val_check_interval


    print(f'+++++++++++++++++++train with gradient clip value {args.gradient_clip_val}')
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        max_steps=args.num_steps,
        devices=args.ngpus,
        num_nodes=args.num_nodes,
        gradient_clip_val = args.gradient_clip_val,
        # accelerator=args.distributed_backend,
        accelerator='gpu',
        default_root_dir=args.log_dir,
        auto_lr_find=False,
        resume_from_checkpoint=args.load_model,
        callbacks=[early_stopping, checkpoint_callback],
        logger=[tb_logger, csv_logger, wandb_logger],
        reload_dataloaders_every_n_epochs=1,
        # reload_dataloaders_every_epoch=False,
        precision=args.precision,
        strategy=ddp_plugin,
        num_sanity_val_steps=2,
        # val_check_interval=val_check_interval,
        check_val_every_n_epoch=args.test_interval,
        # resume_from_checkpoint='/data2/SKData/Pretraining-Denoising/logs/f2d_md17_sample128_frad_multask_Areg_omit_loss_nan_gcv1_benzene/step=233239-epoch=1959-val_loss=0.1266-test_loss=nan-train_per_step=0.0022.ckpt',
        # plugins=[ddp_plugin],
    )

    trainer.fit(model, data)

    # run test set after completing the fit
    # trainer.test()


if __name__ == "__main__":
    main()