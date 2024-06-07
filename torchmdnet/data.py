from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_warn
from torchmdnet import datasets
from torchmdnet.utils import make_splits, MissingEnergyException, DataLoaderMasking
from torch_scatter import scatter
import os
import numpy as np

class DataModule(LightningDataModule):
    def __init__(self, hparams, dataset=None):
        super(DataModule, self).__init__()
        self._set_hparams(hparams.__dict__ if hasattr(hparams, "__dict__") else hparams)
        # self.hparams = hparams.__dict__ if hasattr(hparams, "__dict__") else hparams
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = dataset

        self.mask_atom = hparams.mask_atom

    def setup(self, stage):
        if self.dataset is None:
            if "LBADataset" in self.hparams["dataset"]:
                # special for the atom3d LBA task
                if self.hparams['position_noise_scale'] > 0.:
                    def transform(data):
                            noise = torch.randn_like(data.pos) * self.hparams['position_noise_scale']
                            data.pos_target = noise
                            data.pos = data.pos + noise
                            if self.hparams["prior_model"] == "Atomref":
                                data.y = self.get_energy_data(data)
                            return data
                else:
                    transform = None


                dataset_factory = getattr(datasets, self.hparams["dataset"])

                if self.hparams["dataset"] == 'LBADataset':
                    self.train_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "lba_train.npy"), transform_noise=transform, lp_sep=self.hparams['lp_sep'])
                    self.val_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "lba_valid.npy"), transform_noise=None, lp_sep=self.hparams['lp_sep'])
                    self.test_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "lba_test.npy"), transform_noise=None, lp_sep=self.hparams['lp_sep'])
                else:
                    # self.train_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "train"))
                    # self.val_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "val"))
                    # self.test_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "test"))
                    self.train_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "train_data.pk"))
                    self.val_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "val_data.pk"))
                    self.test_dataset = dataset_factory(os.path.join(self.hparams["dataset_root"], "test_data.pk"))

                # normalize
                if self.hparams["standardize"]:
                    self._standardize()


                return

            elif "oc20_is2re" in self.hparams["dataset"]:
                # special for the oc20 task
                if self.hparams['position_noise_scale'] > 0.:
                    def transform(data):
                            noise = torch.randn_like(data.pos) * self.hparams['position_noise_scale']
                            data.pos_target = noise
                            data.pos = data.pos + noise
                            if self.hparams["prior_model"] == "Atomref":
                                data.y = self.get_energy_data(data)
                            return data
                else:
                    transform = None

                dataset_factory = getattr(datasets, self.hparams["dataset"])

                self.train_dataset = dataset_factory((self.hparams["oc20_train_lmdb"]), transform_noise=transform, lp_sep=self.hparams['lp_sep'])
                self.val_dataset = dataset_factory((self.hparams["oc20_val_lmdb"]), transform_noise=None, lp_sep=self.hparams['lp_sep'])
                self.test_dataset = dataset_factory((self.hparams["oc20_test_lmdb"]), transform_noise=None, lp_sep=self.hparams['lp_sep'])

                # normalize
                if self.hparams["standardize"]:
                    self._standardize()


                return



            if self.hparams["dataset"] == "Custom":
                self.dataset = datasets.Custom(
                    self.hparams["coord_files"],
                    self.hparams["embed_files"],
                    self.hparams["energy_files"],
                    self.hparams["force_files"],
                )
            else:
                if self.hparams['position_noise_scale'] > 0. and 'BIAS' not in self.hparams['dataset'] and 'Dihedral' not in self.hparams['dataset'] and 'QM9A' not in self.hparams['dataset']:
                    def transform(data):
                        noise = torch.randn_like(data.pos) * self.hparams['position_noise_scale']
                        data.pos_target = noise
                        data.pos = data.pos + noise
                        if self.hparams["prior_model"] == "Atomref":
                            data.y = self.get_energy_data(data)
                        return data
                else:
                    transform = None

                if 'BIAS' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], self.hparams['sdf_path'], self.hparams['position_noise_scale'], self.hparams['sample_number'], self.hparams['violate'], dataset_arg=self.hparams["dataset_arg"], transform=t)
                elif 'PCQM4MV2_Force' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], self.hparams['sdf_path'], self.hparams['dihedral_angle_noise_scale'], self.hparams['angle_noise_scale'],  self.hparams['nt_noise_scale'], self.hparams['bond_length_scale'], dataset_arg=self.hparams["dataset_arg"], add_ring_noise=self.hparams['add_ring_noise'], derivative_var=self.hparams['derivative_var'], sample_num=self.hparams['sample_num'], transform=t, add_noise_type=self.hparams['add_noise_type'], use_force_param=self.hparams['use_force_param'], frad_guassian_scale=self.hparams['frad_guassian_scale'])
                elif 'PCQM4MV2_DihedralAll' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], self.hparams['sdf_path'], self.hparams['dihedral_angle_noise_scale'], self.hparams['position_noise_scale'],dataset_arg=self.hparams["dataset_arg"], addh=self.hparams['addh'], transform=t)
                    pass
                elif 'Dihedral2' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], self.hparams['sdf_path'], self.hparams['dihedral_angle_noise_scale'], self.hparams['position_noise_scale'], self.hparams['composition'], self.hparams['decay'], self.hparams['decay_coe'], dataset_arg=self.hparams["dataset_arg"], equilibrium=self.hparams['equilibrium'], eq_weight=self.hparams['eq_weight'], cod_denoise=self.hparams['cod_denoise'], integrate_coord=self.hparams['integrate_coord'], addh=self.hparams['addh'], mask_atom=self.hparams['mask_atom'], mask_ratio=self.hparams['mask_ratio'], bat_noise=self.hparams['bat_noise'], transform=t)
                elif 'DihedralF' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], self.hparams['sdf_path'], self.hparams['dihedral_angle_noise_scale'], self.hparams['position_noise_scale'], self.hparams['composition'], self.hparams['force_field'], self.hparams['pred_noise'], cod_denoise=self.hparams['cod_denoise'], rdkit_conf=self.hparams['rdkit_conf'])
                elif 'Dihedral' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], self.hparams['sdf_path'], self.hparams['dihedral_angle_noise_scale'], self.hparams['position_noise_scale'], self.hparams['composition'], dataset_arg=self.hparams["dataset_arg"], transform=t)
                elif 'QM9A' in self.hparams['dataset'] or 'MD17A' in self.hparams['dataset']:
                    if 'QM9A' in self.hparams['dataset']:
                        if self.hparams["prior_model"] == "Atomref":
                            transform_y = self.get_energy_data
                        else:
                            transform_y = None
                        dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"], transform=None, dihedral_angle_noise_scale=self.hparams['dihedral_angle_noise_scale'], position_noise_scale=self.hparams['position_noise_scale'], composition=self.hparams['composition'], transform_y=transform_y, bat_noise=self.hparams['bat_noise'], torsion_var_r=self.hparams['dihedral_angle_noise_scale'], angle_var=self.hparams['angle_noise_scale'], torsion_var=self.hparams['nt_noise_scale'], bond_var=self.hparams['bond_length_scale'])
                    else: # MD17A
                        dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"], transform=None, dihedral_angle_noise_scale=self.hparams['dihedral_angle_noise_scale'], position_noise_scale=self.hparams['position_noise_scale'], composition=self.hparams['composition'], reverse_half=self.hparams['reverse_half'], addh=self.hparams['addh'], cod_denoise=self.hparams['cod_denoise'], bat_noise=self.hparams['bat_noise'])
                elif 'Pocket' in self.hparams['dataset']:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], self.hparams["dataset_idx"], transform_noise=t)
                elif 'MolNetDataset' in self.hparams['dataset']:
                    # do we really need noise for molenet?
                    # def transform(data):
                    #     noise = torch.randn_like(data.pos) * self.hparams['position_noise_scale']
                    #     data.pos_target = noise
                    #     data.pos = data.pos + noise
                    #     if self.hparams["prior_model"] == "Atomref":
                    #         data.y = self.get_energy_data(data)
                    #     return data
                    data_path = os.path.join(self.hparams['dataset_root'], self.hparams['dataset_arg'])
                    self.train_dataset =  getattr(datasets, self.hparams["dataset"])(f'{data_path}/train.lmdb', self.hparams['dataset_arg'], transform=transform)
                    self.val_dataset = getattr(datasets, self.hparams["dataset"])(f'{data_path}/valid.lmdb', self.hparams['dataset_arg'], transform=transform)
                    self.test_dataset = getattr(datasets, self.hparams["dataset"])(f'{data_path}/test.lmdb', self.hparams['dataset_arg'], transform=transform)


                    # for regression
                    if self.hparams["standardize"]:
                        self._standardize()

                    return
                elif 'DockingDataset' in self.hparams['dataset']:
                    self.train_dataset =  getattr(datasets, self.hparams["dataset"])(f"{self.hparams['dataset_root']}/dataset_v2.csv", self.hparams['dataset_arg'], 'train', f"{self.hparams['dataset_root']}/diversity_molecule_set.pkl", f"{self.hparams['dataset_root']}/docking_id_idx_map.json", transform=transform)
                    self.val_dataset = getattr(datasets, self.hparams["dataset"])(f"{self.hparams['dataset_root']}/dataset_v2.csv", self.hparams['dataset_arg'], 'valid', f"{self.hparams['dataset_root']}/diversity_molecule_set.pkl", f"{self.hparams['dataset_root']}/docking_id_idx_map.json", transform=None)
                    
                    self.test_dataset = getattr(datasets, self.hparams["dataset"])(f"{self.hparams['dataset_root']}/dataset_v2.csv", self.hparams['dataset_arg'], 'test', f"{self.hparams['dataset_root']}/diversity_molecule_set.pkl", f"{self.hparams['dataset_root']}/docking_id_idx_map.json", transform=None)


                    # for regression
                    if self.hparams["standardize"]:
                        self._standardize()

                    return

                else:
                    dataset_factory = lambda t: getattr(datasets, self.hparams["dataset"])(self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"], transform=t)

                # Noisy version of dataset
                self.dataset_maybe_noisy = dataset_factory(transform)
                # Clean version of dataset
                if self.hparams["prior_model"] == "Atomref":
                    def transform_atomref(data):
                        data.y = self.get_energy_data(data)
                        return data
                    self.dataset = dataset_factory(transform_atomref)
                else:
                    self.dataset = dataset_factory(None)

            if self.hparams['use_force_param']: # not all mols have the params
                # load idx:
                ava_idx = np.load('/nfs/SKData/DenoisingData/Param_DB/param_idx.npy')
                # self.dataset = Subset(self.dataset, ava_idx)
                if self.hparams['random_sample'] > 0:
                    ava_idx = np.random.choice(ava_idx, size=self.hparams['random_sample'], replace=False)
                    print(f"===========random sample {self.hparams['random_sample']} from the pre-training dataset")

                self.dataset_maybe_noisy = Subset(self.dataset_maybe_noisy, ava_idx)

            if 'Pocket' in self.hparams['dataset']:
                atom_num_lst = np.load(os.path.join(os.path.dirname(self.hparams["dataset_root"]), 'pocket_atom_num.npy'))
                max_pock_num = self.hparams['pocket_mnum']
                org_data_len = len(self.dataset)
                idx_mask = (atom_num_lst < max_pock_num)
                org_idx = np.array([idx for idx in range(org_data_len)])
                filter_idx = org_idx[idx_mask]
                # simple split:
                self.idx_train = torch.tensor(filter_idx[:-200], dtype=torch.long)
                self.idx_val = torch.tensor(filter_idx[-200:-100], dtype=torch.long)
                self.idx_test = torch.tensor(filter_idx[-100:], dtype=torch.long)

            else:
                if self.hparams['use_force_param']:
                    split_length = len(self.dataset_maybe_noisy)
                else:
                    split_length = len(self.dataset)
                self.idx_train, self.idx_val, self.idx_test = make_splits(
                    split_length,
                    self.hparams["train_size"],
                    self.hparams["val_size"],
                    self.hparams["test_size"],
                    self.hparams["seed"],
                    os.path.join(self.hparams["log_dir"], "splits.npz"),
                    self.hparams["splits"],
                )
            print(
                f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
            )

            self.train_dataset = Subset(self.dataset_maybe_noisy, self.idx_train)

            # If denoising is the only task, test/val datasets are also used for measuring denoising performance.
            if self.hparams['denoising_only']:
                self.val_dataset = Subset(self.dataset_maybe_noisy, self.idx_val)
                self.test_dataset = Subset(self.dataset_maybe_noisy, self.idx_test)
            else:
                self.val_dataset = Subset(self.dataset, self.idx_val)
                self.test_dataset = Subset(self.dataset, self.idx_test)

            if self.hparams["standardize"]:
                self._standardize()

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        # val dataloader
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        if 'MD17' in str(self.dataset):
            if (
                len(self.test_dataset) > 0
                and self.trainer.current_epoch % self.hparams["test_interval"] == 0 and self.trainer.current_epoch != 0
            ):
                loaders.append(self._get_dataloader(self.test_dataset, "test"))
        else:
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = (
            store_dataloader and not self.trainer.reload_dataloaders_every_n_epochs
        )
        if stage in self._saved_dataloaders and store_dataloader:
            # storing the dataloaders like this breaks calls to trainer.reload_train_val_dataloaders
            # but makes it possible that the dataloaders are not recreated on every testing epoch
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False

        if self.mask_atom:
            dl = DataLoaderMasking(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.hparams["num_workers"],
                pin_memory=True,
            )
        else:
            dl = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.hparams["num_workers"],
                pin_memory=True,
                drop_last=bool(self.hparams['drop_last_batch']),
            )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    def get_energy_data(self, data):
        if data.y is None:
            raise MissingEnergyException()


        # remove atomref energies from the target energy
        atomref_energy = self.atomref.squeeze()[data.z].sum()
        return (data.y.squeeze() - atomref_energy).unsqueeze(dim=0).unsqueeze(dim=1)


    def _standardize(self):
        def get_energy(batch, atomref):
            if batch.y is None:
                raise MissingEnergyException()

            if atomref is None:
                return batch.y.clone()

            # remove atomref energies from the target energy
            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        def get_force(batch):
            return batch.dy.clone()

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            # only remove atomref energies if the atomref prior is used
            # atomref = self.atomref if self.hparams["prior_model"] == "Atomref" else None
            atomref = None
            # extract energies from the data
            ys = torch.cat([get_energy(batch, atomref) for batch in data])
            if 'dy' in self.train_dataset[0].keys:
                dys = torch.cat([get_force(batch) for batch in data])
        except MissingEnergyException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return

        # compute mean and standard deviation
        if 'dy' in self.train_dataset[0].keys:
            self._mean = [ys.mean(dim=0), dys.mean(dim=0)]
            self._std = [ys.std(dim=0), dys.std(dim=0)]
        else:
            if self.train_dataset[0].y.shape[0] == 5: # docking dataset
                ys = ys.reshape(-1, 5)
                self._mean = ys.mean(dim=0)
                self._std = ys.std(dim=0)
            else:
                self._mean = ys.mean(dim=0)
                self._std = ys.std(dim=0)
