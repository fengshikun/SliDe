import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, _LRScheduler
from torch.nn.functional import mse_loss, l1_loss, smooth_l1_loss

from pytorch_lightning import LightningModule
from torchmdnet.models.model import create_model, load_model
from torsion_utils import getAngle_torch, wiki_dihedral_torch

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from torch_scatter import scatter
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error
import math
from torch import nn
import copy


class MLP(torch.nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, num_layers, normalization=None
    ):
        super().__init__()
        assert num_layers >= 0, "negative layers?!?"
        if normalization is not None:
            assert callable(normalization), "normalization must be callable"

        if num_layers == 0:
            self.net = torch.nn.Identity()
            return

        if num_layers == 1:
            self.net = torch.nn.Linear(input_dim, output_dim)
            return

        linear_net = torch.nn.Linear

        layers = []
        prev_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(linear_net(prev_dim, hidden_dim))
            if normalization is not None:
                layers.append(normalization())
            layers.append(torch.nn.SiLU())
            # layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim

        layers.append(torch.nn.Linear(hidden_dim, output_dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)




class LNNP(LightningModule):
    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super(LNNP, self).__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams, mean=mean, std=std)
        elif self.hparams.pretrained_model:
            self.model = load_model(self.hparams.pretrained_model, args=self.hparams, mean=mean, std=std)
        else:
            self.model = create_model(self.hparams, prior_model, mean, std)

        if self.hparams['output_model_noise'] == 'VectorOutput3':
            # queue
            self.mom_encoder = copy.deepcopy(self.model)
            self.prediction_head = MLP(
                self.hparams['embedding_dimension'],
                self.hparams['embedding_dimension'],
                self.hparams['embedding_dimension'],
                2,
            )
            
            # queue
            # create the queue
            self.register_buffer("queue", torch.randn(self.hparams['embedding_dimension'], self.hparams['K']))
            self.queue = torch.nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long)) 

        # initialize exponential smoothing
        self.ema = None
        self._reset_ema_dict()

        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()

        # seperate noisy node and finetune target
        self.sep_noisy_node = self.hparams.sep_noisy_node
        self.train_loss_type = self.hparams.train_loss_type

        if self.hparams.mask_atom:
            self.criterion = torch.nn.CrossEntropyLoss()

        self.bond_length_scale = self.hparams.bond_length_scale
        self.derivative_var = self.hparams.derivative_var

        if self.bond_length_scale > 0:
            pass

        # special for ATOM3D LBA dataset
        self.dataset = self.hparams['dataset'] # LBADataset


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        m = 0.999 # 
        for param_q, param_k in zip(self.model.parameters(), self.mom_encoder.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)
            
    
    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = self.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams['K'] % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.K  # move pointer

        self.queue_ptr[0] = ptr
    
    
    def _get_contrastive_predictions(self, q, k):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)


        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        return logits, labels

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.lr_schedule == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, self.hparams.lr_cosine_length)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.hparams.lr_schedule == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        elif self.hparams.lr_schedule == 'cosine_warmup':
            scheduler = CustomScheduler(optimizer=optimizer, max_lr=self.hparams.lr, min_lr=self.hparams.lr_min, iters_per_epoch=len(self.trainer._data_connector._train_dataloader_source.dataloader()), num_epochs=self.hparams.num_epochs)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise ValueError(f"Unknown lr_schedule: {self.hparams.lr_schedule}")
        return [optimizer], [lr_scheduler]

    def forward(self, z, pos, batch=None, batch_org=None, frad_pos=False):
        return self.model(z, pos, batch=batch, batch_org=batch_org, frad_pos=frad_pos)

    def training_step(self, batch, batch_idx):
        if self.train_loss_type == 'smooth_l1_loss':
            return self.step(batch, smooth_l1_loss, 'train')
        return self.step(batch, mse_loss, "train")

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            return self.step(batch, l1_loss, "val")
            # validation step
            return self.step(batch, mse_loss, "val")
        # test step
        return self.step(batch, l1_loss, "test")

    def test_step(self, batch, batch_idx):
        # res = self.step(batch, l1_loss, "test")
        # print(res)
        # return res
        return self.step(batch, l1_loss, "test")


    def process_batch_idx(self, batch):
        # process the idx of bond_target, angle_target and dihedral_target.
        batch_info = batch['batch']
        batch_num = batch._num_graphs

        slice_dict = batch._slice_dict
        bond_target_indx = slice_dict['bond_target']
        angle_target_indx = slice_dict['angle_target']
        dihedral_target_indx = slice_dict['dihedral_target']
        rotate_dihedral_target_indx = slice_dict['rotate_dihedral_target']
        for i in range(batch_num):
            cur_num = slice_dict['pos'][i] # add to bond idx

            batch.bond_target[bond_target_indx[i]:bond_target_indx[i+1]][:, :2] += cur_num
            batch.angle_target[angle_target_indx[i]:angle_target_indx[i+1]][:, :3] += cur_num
            #NOTE: omit the dihedral
            batch.dihedral_target[dihedral_target_indx[i]:dihedral_target_indx[i+1]][:, :4] += cur_num
            # if 'rotate_dihedral_target' in batch.keys:
            #     batch.rotate_dihedral_target[rotate_dihedral_target_indx[i]:rotate_dihedral_target_indx[i+1]][:, :4] += cur_num

    def molnet_loss(self, pred, target, reduction=True):
        if self.hparams.is_cls:
            if self.hparams.cls_number > 1:
                is_valid = target > -0.5
                # multi task
                valid_pred = pred[is_valid]
                valid_target = target[is_valid].to(torch.float)
                loss = F.binary_cross_entropy_with_logits(
                        valid_pred,
                        valid_target,
                        reduction="sum" if reduction else "none",
                    )
                return loss
            else: # single binary classification such like bbbp
                lprobs = F.log_softmax(pred, dim=-1)
                lprobs = lprobs.view(-1, lprobs.size(-1))
                targets = target.view(-1)
                loss = F.nll_loss(
                    lprobs,
                    targets,
                    reduction="sum" if reduction else "none",
                )
                return loss
        else:
            loss = F.mse_loss(
                pred.squeeze(),
                target.squeeze(),
                reduction="sum" if reduction else "none",
            )
            return loss

    def multi_target_auc(self, y_pred, y_true):
        if self.hparams.is_cls:
            if self.hparams.cls_number > 1:

                agg_auc_list = []
                for i in range(y_true.shape[1]):
                # AUC is only defined when there is at least one positive data.
                    if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                        # ignore nan values
                        is_labeled = y_true[:, i] > -0.5
                        agg_auc_list.append(
                            roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                        )
                if len(agg_auc_list) < y_true.shape[1]:
                    print("Some target is missing!")
                if len(agg_auc_list) == 0:
                    raise RuntimeError(
                        "No positively labeled data available. Cannot compute Average Precision."
                    )
                agg_auc = sum(agg_auc_list) / len(agg_auc_list)
                return agg_auc
            else:
                probs = y_pred[:, 1]
                agg_auc = roc_auc_score(y_true.flatten(), probs)
                return agg_auc
        else:
            pass
        
    

    def step(self, batch, loss_fn, stage):
        if self.dataset == 'MolNetDataset':
            loss_fn = self.molnet_loss
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            # TODO: the model doesn't necessarily need to return a derivative once
            # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)
            if stage == 'test' and 'org_pos' in batch.keys:
                pred, noise_pred, deriv = self(batch.z, batch.org_pos, batch.batch) # use the org pos
            else:
                if self.sep_noisy_node:
                    pred, _, deriv = self(batch.z, batch.org_pos, batch.batch)
                    _, noise_pred, _ = self(batch.z, batch.pos, batch.batch)
                else:

                    # if self.bond_length_scale > 0:
                    if 'bond_target' in batch.keys:
                        self.process_batch_idx(batch)
                        if 'pos_frad' in batch.keys:
                            _, frad_noise_pred, _ = self(batch.z, batch.pos_frad, batch.batch, batch_org=batch, frad_pos=True)
                        pred, noise_pred, deriv = self(batch.z, batch.pos, batch.batch, batch_org=batch)
                    elif 'LBADataset' in self.dataset or 'type_mask' in batch.keys:
                        pred, noise_pred, deriv = self(batch.z, batch.pos, batch.batch, batch_org=batch)
                    elif 'ScalarMolInject' in self.hparams.output_model:
                        pred, noise_pred, deriv = self(batch.z, batch.pos, batch.batch, batch_org=batch)
                    else:
                        pred, noise_pred, deriv = self(batch.z, batch.pos, batch.batch)

        
        
        denoising_is_on = ("pos_target" in batch or "bond_target" in batch) and (self.hparams.denoising_weight > 0) and (noise_pred is not None)


        loss_y, loss_dy, loss_pos, mask_atom_loss = 0, 0, 0, 0

        if len(noise_pred) == 3:
            noise_pred, ctl_pred, org_x = noise_pred
            
            # ctl_pred
            q = self.prediction_head(ctl_pred)
            
            
            # 
            _, noise_pred2, _ = self.mom_encoder(batch.z, batch.pos_target + batch.pos, batch.batch)
            _, k, _ = noise_pred2
            
            
            # _, noise_pred2, _ = self(batch.z, batch.pos_target, batch.batch)
            # _, _, ctl_target = noise_pred2
            # scatter first and mse loss
            # ctl_pred = scatter(ctl_pred, batch.batch, dim=0, reduce=self.hparams['reduce_op'])
            # ctl_target = scatter(ctl_target, batch.batch, dim=0, reduce=self.hparams['reduce_op']).detach()
            
            q = scatter(q, batch.batch, dim=0, reduce=self.hparams['reduce_op'])
            k = scatter(k, batch.batch, dim=0, reduce=self.hparams['reduce_op']).detach()
            
            
            q = torch.nn.functional.normalize(q, dim=1)
            k = torch.nn.functional.normalize(k, dim=1)
            
            
            
            logits, labels = self._get_contrastive_predictions(q, k)
            
            
            # ctl_3d_loss = nn.functional.mse_loss(ctl_pred, ctl_target)
            ctl_3d_loss = F.cross_entropy(logits / self.hparams['T'], labels) # temperature
            loss_y = ctl_3d_loss.mean()
            self.losses[stage + "_ctl_3d_loss"].append(ctl_3d_loss.detach())
            
            
            with torch.no_grad():
                self._momentum_update_key_encoder()
            # dequeue and enqueue
            self._dequeue_and_enqueue(k)


        # check whether mask, only in pretraining, deriv is logits
        if self.hparams.mask_atom:
            mask_indices = batch['masked_atom_indices']
            mask_logits = deriv[mask_indices]
            mask_atom_loss = self.criterion(mask_logits, batch.mask_node_label)
            self.losses[stage + "_mask_atom_loss"].append(mask_atom_loss.detach())

        if self.hparams.derivative:
            if "y" not in batch:
                # "use" both outputs of the model's forward function but discard the first
                # to only use the derivative and avoid 'Expected to have finished reduction
                # in the prior iteration before starting a new one.', which otherwise get's
                # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
                deriv = deriv + pred.sum() * 0

            # force/derivative loss

            # if noise_pred is not None:
            #     deriv = noise_pred

            if stage in ["train", "val"]:
                loss_dy = loss_fn(deriv, batch.dy)
            else:
                loss_dy = loss_fn(deriv, batch.dy, reduction='none')
                loss_dy = scatter(loss_dy, batch.batch, dim=0, reduce='mean')
                loss_dy = loss_dy.mean(dim=1)

            if stage in ["train", "val"] and self.hparams.ema_alpha_dy < 1:
                if self.ema[stage + "_dy"] is None:
                    self.ema[stage + "_dy"] = loss_dy.detach()
                # apply exponential smoothing over batches to dy
                loss_dy = (
                    self.hparams.ema_alpha_dy * loss_dy
                    + (1 - self.hparams.ema_alpha_dy) * self.ema[stage + "_dy"]
                )
                self.ema[stage + "_dy"] = loss_dy.detach()

            if self.hparams.force_weight > 0:
                self.losses[stage + "_dy"].append(loss_dy.detach())

        if "y" in batch:
            if (noise_pred is not None) and not denoising_is_on:
                # "use" both outputs of the model's forward (see comment above).
                pred = pred + noise_pred.sum() * 0

            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)

            # if self.hparams["prior_model"] == "Atomref":
            #     batch.y = self.get_energy(batch)

            if torch.isnan(pred).sum():
                print('pred nan happends')
            # energy/prediction loss
            if stage in ["train", "val"]:
                loss_y = loss_fn(pred, batch.y)
            else:
                loss_y = loss_fn(pred, batch.y, reduction='none')

            # if atom3d lba
            if ('LBADataset' in self.dataset or 'MolNetDataset' in self.dataset or 'DockingDataset' in self.dataset) and stage in ['val', 'test']:
                if stage == 'test':
                    self.losses['lba_pred_org'].append(pred.detach())
                    self.losses['lba_y_org'].append(batch.y)
                else:
                    self.losses['lba_pred_org_val'].append(pred.detach())
                    self.losses['lba_y_org_val'].append(batch.y)




            if torch.isnan(loss_y).sum():
                print('loss nan happens')

            if stage in ["train", "val"] and self.hparams.ema_alpha_y < 1:
                if self.ema[stage + "_y"] is None:
                    self.ema[stage + "_y"] = loss_y.detach()
                # apply exponential smoothing over batches to y
                loss_y = (
                    self.hparams.ema_alpha_y * loss_y
                    + (1 - self.hparams.ema_alpha_y) * self.ema[stage + "_y"]
                )
                self.ema[stage + "_y"] = loss_y.detach()

            if self.hparams.energy_weight > 0:
                self.losses[stage + "_y"].append(loss_y.detach())
                if stage != 'train':
                    loss_y = loss_y.mean()


            # if oc20
            if 'oc20_is2re' in self.hparams['dataset']:
                if stage == "val":
                    self.losses['val_y_label'].append(batch.y)
                    self.losses['val_y_pred'].append(pred.detach())
                elif stage == "test":
                    self.losses['test_y_label'].append(batch.y)
                    self.losses['test_y_pred'].append(pred.detach())


        if denoising_is_on:
            if "y" not in batch:
                if isinstance(noise_pred, list): # bond angle dihedral
                    noise_pred = [ele + pred.sum() * 0 for ele in noise_pred]
                else:
                # "use" both outputs of the model's forward (see comment above).
                    noise_pred = noise_pred + pred.sum() * 0

            def weighted_mse_loss(input, target, weight):
                return (weight.reshape(-1, 1).repeat((1, 3)) * (input - target) ** 2).mean()
            def mse_loss(input, target):
                return ((input - target) ** 2).mean()

            def mse_loss_torsion(input, target):
                diff_torsion = input - target

                # diff_torsion[diff_torsion < -180] += 360
                # diff_torsion[diff_torsion > 180] -= 360
                mask = (diff_torsion > -180) & (diff_torsion < 180)
                # diff_torsion[diff_torsion < -180] = 0
                # diff_torsion[diff_torsion > 180] = 0

                diff_torsion = diff_torsion[mask]
                # less_neg_180 = (target < -180)
                # target[less_neg_180] += 360
                # target[less_neg_180] = -target[less_neg_180]

                # big_than_180 = (target > 180)
                # target[big_than_180] -= 360
                # target[big_than_180] = -target[big_than_180]

                # target[target > 180] -= 360
                return (diff_torsion ** 2).mean()

            if 'wg' in batch.keys:
                loss_fn = weighted_mse_loss
                wt = batch['w1'].sum() / batch['idx'].shape[0]
                weights = batch['wg'] / wt
            else:
                loss_fn = mse_loss
            if len(noise_pred) == 7:
                coord_noise_pred, bond_noise_pred, angle_noise_pred, dihedral_noise_pred, bond_valid_idx, angle_valid_idx, dihedral_valid_idx = noise_pred

                normalized_pos_target = self.model.pos_normalizer(batch.pos_target)
                loss_pos = loss_fn(coord_noise_pred, normalized_pos_target)
                self.losses[stage + "_pos"].append(loss_pos.detach())

                normalized_bond_target = self.model.bond_pos_normalizer(batch.bond_target[:,-1][bond_valid_idx.to(torch.long)])
                normalized_angle_target = self.model.angle_pos_normalizer(batch.angle_target[:,-1][angle_valid_idx.to(torch.long)])
                normalized_dihedral_target = self.model.dihedral_pos_normalizer(batch.dihedral_target[:,-1][dihedral_valid_idx.to(torch.long)])
                loss_bond = loss_fn(bond_noise_pred, normalized_bond_target)
                loss_angle = loss_fn(angle_noise_pred, normalized_angle_target)
                loss_dihedral = loss_fn(dihedral_noise_pred, normalized_dihedral_target)
                self.losses[stage + "_bond"].append(loss_bond.detach())
                self.losses[stage + "_angle"].append(loss_angle.detach())
                self.losses[stage + "_dihedral"].append(loss_dihedral.detach())

                loss_pos = loss_pos + loss_bond + loss_angle + loss_dihedral
                if loss_pos.isnan().sum().item():
                    print('loss nan!!!')
            elif self.model.bond_pos_normalizer is not None:
                # bond, angle, dihedral
                normalized_bond_target = self.model.bond_pos_normalizer(batch.bond_target[:,-1])
                normalized_angle_target = self.model.angle_pos_normalizer(batch.angle_target[:,-1])
                # normalized_dihedral_target = self.model.dihedral_pos_normalizer(batch.dihedral_target[:,-1])
                # normalized_rotate_dihedral_target = self.model.rotate_dihedral_pos_normalizer(batch.rotate_dihedral_target[:,-1])

                # if len(noise_pred) == 8:
                #     normalized_bond_target = normalized_bond_target[noise_pred[4].to(torch.long)]
                #     normalized_angle_target = normalized_angle_target[noise_pred[5].to(torch.long)]
                #     normalized_dihedral_target = normalized_dihedral_target[noise_pred[6].to(torch.long)]
                #     normalized_rotate_dihedral_target = normalized_rotate_dihedral_target[noise_pred[7].to(torch.long)]
                # if len(noise_pred) == 5:
                #     normalized_bond_target = normalized_bond_target[noise_pred[3].to(torch.long)]
                #     normalized_angle_target = normalized_angle_target[noise_pred[4].to(torch.long)]

                if 'reg_value_lst' in batch.keys:
                    assert 'noise_target_lst' in batch.keys
                    # for each sample


                    batch_size = batch.batch.max().item() + 1
                    if 'ff_learn' in batch.keys:
                        # collect result
                        if 'pos_frad' in batch.keys:
                            nan_indx = batch.pos_target.isnan().sum(axis=1)
                            de_nan_indx = ~(nan_indx.to(torch.bool))
                            # batch.pos_target = batch.pos_target[de_nan_indx]
                            pos_target_denan = batch.pos_target[de_nan_indx]
                            if isinstance(noise_pred, list):
                                pos_pred = frad_noise_pred[0]
                            else:
                                pos_pred = frad_noise_pred

                            # for debug
                            # if self.trainer.global_step < 5 and self.trainer.local_rank == 0:
                            #     pos_pred[:10,:] = float('nan')

                            pos_pred_denan = pos_pred[de_nan_indx]

                            normalized_pos_target = self.model.pos_normalizer(pos_target_denan)
                            loss_pos = loss_fn(pos_pred_denan, normalized_pos_target)
                            self.losses[stage + "_frad_pos"].append(loss_pos.detach())


                            # calculate the ff_loss
                            # ff_learn_pos = np.concatenate(batch['ff_learn'], axis=0)
                            # ff_learn_pos = torch.tensor(ff_learn_pos).to(pos_pred.device)
                            ff_learn_pos_norm = self.model.ff_normalizer(batch['ff_learn'])

                            if isinstance(noise_pred, list):
                                ff_pred = noise_pred[0]
                            else:
                                ff_pred = noise_pred
                            # ff_pred = frad_noise_pred[0]
                            loss_ff = loss_fn(ff_pred, ff_learn_pos_norm)
                            self.losses[stage + "_ffreg"].append(loss_ff.detach())
                            loss_pos += loss_ff



                            if self.trainer.global_step > 2000 and loss_pos > 1:
                                loss_pos = loss_pos * 0
                                # loss_pos = torch.zeros_like(loss_ff, dtype=torch.float32, requires_grad=True)
                                print('loss greater than 1 after 2000 setp, omit this batch')

                            gather_loss = self.all_gather(loss_pos)
                            # if any(torch.isnan(loss) for loss in gather_loss):
                            if torch.isnan(gather_loss).sum():
                                print(f'loss nan happens, omit this batch (all card) at rank {self.trainer.local_rank}')
                                return None  # skip training step

                            # if torch.isnan(loss_pos).sum():
                            #     # loss_pos = loss_pos * 0
                            #     loss_pos = torch.zeros_like(loss_ff, dtype=torch.float32, requires_grad=True)
                            #     print('loss nan happens, omit this batch')
                            # loss_pos = torch.zeros_like(loss_ff, dtype=torch.float32, requires_grad=True)
                            # loss_pos = loss_pos * 0
                        else:

                            # NOTE debug without the denan logic
                            if isinstance(noise_pred, list):
                                pos_pred = noise_pred[0]
                            else:
                                pos_pred = noise_pred
                            # ff_learn_pos = np.concatenate(batch['ff_learn'], axis=0)
                            # ff_learn_pos = torch.tensor(ff_learn_pos).to(pos_pred.device)
                            ff_learn_pos_norm = self.model.pos_normalizer(batch['ff_learn'])

                            loss_pos = loss_fn(pos_pred, ff_learn_pos_norm)
                            self.losses[stage + "_ffreg"].append(loss_pos.detach())
                    else:

                        batch_idx = batch.batch
                        loss_pos = 0
                        if isinstance(noise_pred, list):
                            pos_pred = noise_pred[0]
                        else:
                            pos_pred = noise_pred
                        for bidx in range(batch_size):
                            pos_pred_sample = pos_pred[batch_idx==bidx]
                            sample_noise_target_lst = torch.tensor(batch.noise_target_lst[bidx]).cuda().detach()
                            pred_sample = pos_pred_sample * sample_noise_target_lst
                            pred_sample_values = pred_sample.sum(dim=2).sum(dim=1)
                            target_values = torch.tensor(batch['reg_value_lst'][bidx]).cuda().detach()

                            ntarget_values = self.model.reg_dev_normalizer(target_values)
                            sample_loss = loss_fn(pred_sample_values, ntarget_values)
                            loss_pos += sample_loss
                        loss_pos = loss_pos / batch_size
                else:
                    # batch pos_target may contains nan
                    nan_indx = batch.pos_target.isnan().sum(axis=1)
                    de_nan_indx = ~(nan_indx.to(torch.bool))
                    # batch.pos_target = batch.pos_target[de_nan_indx]
                    pos_target_denan = batch.pos_target[de_nan_indx]
                    pos_pred = noise_pred[0]
                    pos_pred_denan = pos_pred[de_nan_indx]

                    normalized_pos_target = self.model.pos_normalizer(pos_target_denan)
                    loss_pos = loss_fn(pos_pred_denan, normalized_pos_target)
                    self.losses[stage + "_pos"].append(loss_pos.detach())

                    loss_bond = loss_fn(noise_pred[1], normalized_bond_target)
                    # loss_angle = loss_fn(noise_pred[2], normalized_angle_target)

                    true_pos_pred = pos_pred * self.model.pos_normalizer.std + self.model.pos_normalizer.mean
                    # loss_angle = loss_fn(noise_pred[1], batch.angle_target[:,-1])
                    angle_idx = batch.angle_target[:,:3].to(torch.long)
                    angle_label = batch.angle_target[:,3]
                    angle_noise = getAngle_torch(batch.pos, angle_idx)
                    angle_org = getAngle_torch(batch.pos - batch.pos_target, angle_idx)
                    # check angle_org + angle_label = angle_noise
                    angle_pred = getAngle_torch(batch.pos - true_pos_pred, angle_idx)

                    loss_angle = loss_fn(angle_pred, angle_org)

                    torsion_idx = batch.dihedral_target[:,:4].to(torch.long)
                    torsion_label = batch.dihedral_target[:,4]
                    torsion_noise = wiki_dihedral_torch(batch.pos, torsion_idx)
                    torsion_org = wiki_dihedral_torch(batch.pos - batch.pos_target, torsion_idx)
                    # check torsion_org + torsion_label = torsion_noise
                    torsion_pred = wiki_dihedral_torch(batch.pos - true_pos_pred, torsion_idx)

                    loss_torsion = mse_loss_torsion(torsion_pred, torsion_org)

                    # loss_dihedral = loss_fn(noise_pred[2], normalized_dihedral_target)
                    # # loss_dihedral = loss_fn(noise_pred[2], batch.dihedral_target[:,-1])

                    # # loss_rotate_dihedral = loss_fn(noise_pred[3], batch.rotate_dihedral_target[:,-1])
                    # loss_rotate_dihedral = loss_fn(noise_pred[3], normalized_rotate_dihedral_target)


                    self.losses[stage + "_bond"].append(loss_bond.detach())
                    # self.losses[stage + "_angle"].append(loss_angle.detach() / (self.model.angle_pos_normalizer.std ** 2))
                    self.losses[stage + "_angle"].append(loss_angle.detach())


                    self.losses[stage + "_torsion"].append(loss_torsion.detach())
                    # self.losses[stage + "_dihedral"].append(loss_dihedral.detach())
                    # # self.losses[stage + "_dihedral"].append(loss_dihedral.detach() / (self.model.dihedral_pos_normalizer.std ** 2))
                    # self.losses[stage + "_rotate_dihedral"].append(loss_rotate_dihedral.detach())
                    # self.losses[stage + "_rotate_dihedral"].append(loss_rotate_dihedral.detach() / (self.model.rotate_dihedral_pos_normalizer.std ** 2))
                    # loss_pos = loss_bond + loss_angle + loss_dihedral + loss_rotate_dihedral
                    if self.current_epoch > 0: # start from 0
                        loss_pos = loss_pos + loss_bond
                        # + 0.01 * loss_angle + 0.0001 * loss_torsion
                    else:
                        loss_pos = loss_pos + loss_bond
                    # + loss_angle
                    if loss_pos.isnan().sum().item():
                        print('loss nan!!!')

            elif self.model.pos_normalizer is not None:
                normalized_pos_target = self.model.pos_normalizer(batch.pos_target)
                if 'wg'in batch.keys:
                    loss_pos = loss_fn(noise_pred, normalized_pos_target, weights)
                else:
                    loss_pos = loss_fn(noise_pred, normalized_pos_target)
                self.losses[stage + "_pos"].append(loss_pos.detach())
            else:
                if 'wg'in batch.keys:
                    loss_pos = loss_fn(noise_pred, normalized_pos_target, weights)
                else:
                    loss_pos = loss_fn(noise_pred, normalized_pos_target)
            # loss_pos = loss_fn(noise_pred, normalized_pos_target)
                self.losses[stage + "_pos"].append(loss_pos.detach())

        # total loss
        loss = loss_y * self.hparams.energy_weight + loss_dy * self.hparams.force_weight + loss_pos * self.hparams.denoising_weight + mask_atom_loss

        self.losses[stage].append(loss.detach())

        # Frequent per-batch logging for training
        if stage == 'train':
            train_metrics = {k + "_per_step": v[-1] for k, v in self.losses.items() if (k.startswith("train") and len(v) > 0)}
            train_metrics['lr_per_step'] = self.trainer.optimizers[0].param_groups[0]["lr"]
            train_metrics['step'] = self.trainer.global_step
            train_metrics['batch_pos_mean'] = batch.pos.mean().item()
            self.log_dict(train_metrics, sync_dist=True)

        if torch.isnan(loss).sum():
            print('loss nan happens')

        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def training_epoch_end(self, training_step_outputs):
        dm = self.trainer.datamodule
        if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
            should_reset = (
                self.current_epoch % self.hparams.test_interval == 0
                or (self.current_epoch - 1) % self.hparams.test_interval == 0
            )
            if should_reset:
                # reset validation dataloaders before and after testing epoch, which is faster
                # than skipping test validation steps by returning None
                self.trainer.reset_val_dataloader(self)
                
        loss_epoch_sum = 0
        for i in training_step_outputs:
            loss_epoch_sum += i['loss'].item()
        loss_epoch_mean = torch.tensor(loss_epoch_sum / len(training_step_outputs)).type_as(training_step_outputs[0]['loss'])
        self.trainer.callback_metrics['train_loss_epoch'] = loss_epoch_mean
        
        
    def test_epoch_end(self, outputs):
        result_dict = {}
        if len(self.losses["test_y"]) > 0:
            if not self.losses['test_y'][0].dim():
                result_dict["test_loss_y"] = torch.stack(
                    self.losses["test_y"]
                ).mean()
            else:
                result_dict["test_loss_y"] = torch.cat(
                    self.losses["test_y"]
                ).mean()

        if len(self.losses["lba_pred_org"]) > 0:
            lba_pred_org = torch.cat(self.losses["lba_pred_org"])
            lba_y_org = torch.cat(self.losses["lba_y_org"])
            if 'MolNetDataset' in self.dataset:
                y_true = lba_y_org.cpu().numpy()
                y_pred = lba_pred_org.cpu().numpy()
                # np.save('y_pred4.npy', y_pred)
                if self.hparams.is_cls:
                    agg_auc = self.multi_target_auc(y_pred, y_true)
                    result_dict['test_roc_auc'] = agg_auc
                    print(f'test auc is {agg_auc}')
                else:
                    agg_rmse = mean_squared_error(y_pred, y_true)
                    result_dict['test_rmse'] = agg_rmse
                    print(f'test rmse is {agg_rmse}')
            
            elif "DockingDataset" in self.dataset:
                spearman_corr_lst, rmse_lst, pearson_corr_lst = self.compute_metrics_dockingdataset(lba_pred_org, lba_y_org)
                sub_tasks = ['docking_score', 'emodel_score', 'hbond_score', 'polar_score', 'coulombic_score']
                for i, sub_task in enumerate(sub_tasks):
                    result_dict[f'test_rmse_{sub_task}'] = rmse_lst[i]
                    result_dict[f'test_spearman_{sub_task}'] = spearman_corr_lst[i]
                    result_dict[f'test_pearson_{sub_task}'] = pearson_corr_lst[i]

            else:
                spearman_corr, rmse, pearson_corr = self.compute_metrics_lba(lba_pred_org, lba_y_org)
                result_dict["test_rmse_y"] = rmse
                result_dict["test_spearman"] = spearman_corr
                result_dict["test_pearson"] = pearson_corr
        return result_dict

    # TODO(shehzaidi): clean up this function, redundant logging if dy loss exists.
    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {
                "epoch": self.current_epoch,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
                # "val_loss": torch.cat(self.losses["val"]).mean(),
            }

            # add test loss if available
            if len(self.losses["test"]) > 0:
                if not len(self.losses["test"][0].shape):
                    result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()
                else:
                    result_dict["test_loss"] = torch.cat(self.losses["test"]).mean()

            # if prediction and derivative are present, also log them separately
            if len(self.losses["train_y"]) > 0 and len(self.losses["train_dy"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
                result_dict["train_loss_dy"] = torch.stack(
                    self.losses["train_dy"]
                ).mean()
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
                result_dict["val_loss_dy"] = torch.stack(self.losses["val_dy"]).mean()

                if len(self.losses["test"]) > 0:
                    result_dict["test_loss_y"] = torch.cat(
                        self.losses["test_y"], dim=0
                    ).nanmean()
                    result_dict["test_loss_dy"] = torch.cat(
                        self.losses["test_dy"], dim=0
                    ).nanmean()
                    print(f'test y {result_dict["test_loss_y"]}  test dy {result_dict["test_loss_dy"]}')

            if len(self.losses["train_y"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
            if len(self.losses['val_y']) > 0:
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
            #   result_dict["val_loss_y"] = torch.concat(self.losses["val_y"]).mean()
            # if len(self.losses["test_y"]) > 0:
            #     result_dict["test_loss_y"] = torch.stack(self.losses["test_y"]).mean()
            if len(self.losses["test_y"]) > 0:
                if not self.losses['test_y'][0].dim():
                    result_dict["test_loss_y"] = torch.stack(
                        self.losses["test_y"]
                    ).mean()
                else:
                    result_dict["test_loss_y"] = torch.cat(
                        self.losses["test_y"]
                    ).mean()
                # result_dict["test_loss_y"] = torch.cat(
                #     self.losses["test_y"]
                # ).mean()
            
            # for aotom 3d
            if len(self.losses["lba_pred_org"]) > 0:

                lba_pred_org = torch.cat(self.losses["lba_pred_org"])
                lba_y_org = torch.cat(self.losses["lba_y_org"])
                if 'MolNetDataset' in self.dataset:
                    y_true = lba_y_org.cpu().numpy()
                    y_pred = lba_pred_org.cpu().numpy()
                    if self.hparams.is_cls:
                        agg_auc = self.multi_target_auc(y_pred, y_true)
                        result_dict['test_roc_auc'] = agg_auc
                        print(f'test auc is {agg_auc}')
                    else:
                        agg_rmse = mean_squared_error(y_pred, y_true)
                        result_dict['test_rmse'] = agg_rmse
                        print(f'test rmse is {agg_rmse}')
                
                elif "DockingDataset" in self.dataset:
                    spearman_corr_lst, rmse_lst, pearson_corr_lst = self.compute_metrics_dockingdataset(lba_pred_org, lba_y_org)
                    sub_tasks = ['docking_score', 'emodel_score', 'hbond_score', 'polar_score', 'coulombic_score']
                    for i, sub_task in enumerate(sub_tasks):
                        result_dict[f'test_rmse_{sub_task}'] = rmse_lst[i]
                        result_dict[f'test_spearman_{sub_task}'] = spearman_corr_lst[i]
                        result_dict[f'test_pearson_{sub_task}'] = pearson_corr_lst[i]

                else:
                    spearman_corr, rmse, pearson_corr = self.compute_metrics_lba(lba_pred_org, lba_y_org)
                    result_dict["test_rmse_y"] = rmse
                    result_dict["test_spearman"] = spearman_corr
                    result_dict["test_pearson"] = pearson_corr

            if len(self.losses["lba_pred_org_val"]) > 0:

                lba_pred_org = torch.cat(self.losses["lba_pred_org_val"])
                lba_y_org = torch.cat(self.losses["lba_y_org_val"])
                
                if 'MolNetDataset' in self.dataset:
                    y_true = lba_y_org.cpu().numpy()
                    y_pred = lba_pred_org.cpu().numpy()
                    if self.hparams.is_cls:
                        agg_auc = self.multi_target_auc(y_pred, y_true)
                        result_dict['val_roc_auc'] = agg_auc
                        print(f'valid auc is {agg_auc}')
                    else:
                        agg_rmse = mean_squared_error(y_pred.squeeze(), y_true.squeeze())
                        result_dict['val_rmse'] = agg_rmse
                        print(f'val rmse is {agg_rmse}')
                elif "DockingDataset" in self.dataset:
                    spearman_corr_lst, rmse_lst, pearson_corr_lst = self.compute_metrics_dockingdataset(lba_pred_org, lba_y_org)
                    sub_tasks = ['docking_score', 'emodel_score', 'hbond_score', 'polar_score', 'coulombic_score']
                    for i, sub_task in enumerate(sub_tasks):
                        result_dict[f'val_rmse_{sub_task}'] = rmse_lst[i]
                        result_dict[f'val_spearman_{sub_task}'] = spearman_corr_lst[i]
                        result_dict[f'val_pearson_{sub_task}'] = pearson_corr_lst[i]
                else:
                    spearman_corr, rmse, pearson_corr = self.compute_metrics_lba(lba_pred_org, lba_y_org)
                    result_dict["val_rmse_y"] = rmse
                    result_dict["val_spearman"] = spearman_corr
                    result_dict["val_pearson"] = pearson_corr

            # for oc20
            if 'oc20_is2re' in self.dataset:
                if len(self.losses["test_y_label"]) > 0:
                    y_label = torch.concat(self.losses["test_y_label"]).reshape(-1)
                    y_pred = torch.concat(self.losses["test_y_pred"]).reshape(-1)
                    mae = abs(y_label - y_pred)
                    ewt = sum(mae < 0.02) / mae.shape[0]
                    result_dict["test_mae"] = mae.mean()
                    result_dict["test_ewt"] = ewt

                if len(self.losses["val_y_label"]) > 0:
                    y_label = torch.concat(self.losses["val_y_label"]).reshape(-1)
                    y_pred = torch.concat(self.losses["val_y_pred"]).reshape(-1)
                    mae = abs(y_label - y_pred)
                    ewt = sum(mae < 0.02) / mae.shape[0]
                    result_dict["val_mae"] = mae.mean()
                    result_dict["val_ewt"] = ewt



            # if denoising is present, also log it
            if len(self.losses["train_pos"]) > 0:
                result_dict["train_loss_pos"] = torch.stack(
                    self.losses["train_pos"]
                ).mean()

            if len(self.losses["val_pos"]) > 0:
                result_dict["val_loss_pos"] = torch.stack(
                    self.losses["val_pos"]
                ).mean()

            if len(self.losses["test_pos"]) > 0:
                result_dict["test_loss_pos"] = torch.stack(
                    self.losses["test_pos"]
                ).mean()

            if "DockingDataset" in self.dataset:
                print(result_dict)
            
            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
            "train_y": [],
            "val_y": [],
            "test_y": [],
            "train_dy": [],
            "val_dy": [],
            "test_dy": [],
            "train_pos": [],
            "val_pos": [],
            "test_pos": [],
            "train_ffreg": [],
            "val_ffreg": [],
            "test_ffreg": [],

            "train_frad_pos": [],
            "val_frad_pos": [],
            "test_frad_pos": [],

            "train_mask_atom_loss": [],
            "val_mask_atom_loss": [],
            "test_mask_atom_loss": [],

            "train_bond": [],
            "val_bond": [],
            "test_bond": [],

            "train_angle": [],
            "val_angle": [],
            "test_angle": [],

            "train_torsion": [],
            "val_torsion": [],
            "test_torsion": [],

            "train_dihedral": [],
            "val_dihedral": [],
            "test_dihedral": [],

            "train_rotate_dihedral": [],
            "val_rotate_dihedral": [],
            "test_rotate_dihedral": [],

            # for lba test
            "lba_pred_org": [],
            "lba_pred_org_val": [],
            "lba_y_org": [],
            "lba_y_org_val": [],
            # "test_rmse": [],
            # "test_spearman_corr": [],
            # "test_pearson_corr": []

            # for molnet test:
            "val_roc_auc": [],
            "test_roc_auc": [],
            "val_rmse": [],
            "test_rmse": [],
            
            "train_ctl_3d_loss": [],
            "val_ctl_3d_loss": [],
            "test_ctl_3d_loss": [],

            # for oc20:
            'val_y_label': [],
            'val_y_pred': [],
            'test_y_label': [],
            'test_y_pred': [],
            
            # for docking dataset

        }

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}


    def compute_metrics_lba(self, predictions, targets):
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        spearman_corr, _ = spearmanr(predictions, targets)
        rmse = np.sqrt(np.mean((predictions-targets)**2))
        # rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
        pearson_corr, _ = pearsonr(predictions.flatten(), targets.flatten())

        return spearman_corr, rmse, pearson_corr
    
    def compute_metrics_dockingdataset(self, predictions, targets):
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        spearman_corr_lst = []
        rmse_lst = []
        pearson_corr_lst = []
        for i in range(predictions.shape[1]):
            spearman_corr, _ = spearmanr(predictions[:,i], targets[:,i])
            rmse = np.sqrt(np.mean((predictions[:,i]-targets[:,i])**2))
            pearson_corr, _ = pearsonr(predictions[:,i], targets[:,i])
            spearman_corr_lst.append(spearman_corr)
            rmse_lst.append(rmse)
            pearson_corr_lst.append(pearson_corr)
        return spearman_corr_lst, rmse_lst, pearson_corr_lst
        

    def test_epoch_end_lba(self, outputs):
        spearman_corr_values = [output['spearman_corr'] for output in outputs]
        rmse_values = [output['rmse'] for output in outputs]
        pearson_corr_values = [output['pearson_corr'] for output in outputs]

        # average
        avg_spearman_corr = torch.tensor(spearman_corr_values).mean()
        avg_rmse = torch.tensor(rmse_values).mean()
        avg_pearson_corr = torch.tensor(pearson_corr_values).mean()
        # print and record the average metric
        # self.log('avg_spearman_corr', avg_spearman_corr, prog_bar=True)
        # self.log('avg_rmse', avg_rmse, prog_bar=True)
        # self.log('avg_pearson_corr', avg_pearson_corr, prog_bar=True)

class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, max_lr, min_lr, iters_per_epoch, num_epochs, last_epoch=-1):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.iters_per_epoch = iters_per_epoch
        self.num_epochs = num_epochs
        self.total_iters = iters_per_epoch * num_epochs
        self.warmup_epoch = 0.3
        self.patience_epoch = 0.7
        super(CustomScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cur_iter = self.last_epoch
        if cur_iter < self.warmup_epoch * self.iters_per_epoch:
            return [self.max_lr * cur_iter / (self.warmup_epoch * self.total_iters) for base_lr in self.base_lrs]
        elif cur_iter < (self.patience_epoch + self.warmup_epoch) * self.iters_per_epoch:
            return [self.max_lr for base_lr in self.base_lrs]
        else:
            prev_iters = (self.patience_epoch + self.warmup_epoch) * self.iters_per_epoch
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (cur_iter - prev_iters) / (self.total_iters - prev_iters)))
            return [lr for base_lr in self.base_lrs]
