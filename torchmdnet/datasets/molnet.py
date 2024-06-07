import pickle
import lmdb
import os
import torch
from functools import lru_cache
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import Dataset
from rdkit.Chem import PeriodicTable


import pandas as pd
import json
# processing the atom type 



period_atomic_number = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
                 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
                 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
                 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
                 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
                 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
                 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
                 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
                 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118, "*": 50}


class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        return data



# for molenet training
# todo know the task number
classification_tasks = ['bbbp', 'bace', 'clintox', 'tox21', 'toxcast', 'sider', 'hiv', 'pcba', 'muv']

class MolNetDataset(Dataset):
    def __init__(self, dataset_path, dataset_name, conf_size=10, transform=None, max_size=512):
        self.dataset = LMDBDataset(dataset_path)
        self.dataset_name = dataset_name
        self.atoms = 'atoms'
        self.coordinates = 'coordinates'
        # self.set_epoch(None)
        
        self.is_train = 'train' in dataset_path
        self.conf_size = conf_size
        self.transform_noise = transform
        self.max_size = max_size
        
        if self.dataset_name in classification_tasks:
            self.classification = True
        else:
            self.classification = False
            
        if self.classification:
            # multitask or single task
            self.cls_number = len(self.dataset[0]['target'])

    # def set_epoch(self, epoch, **unused):
    #     super().set_epoch(epoch)
    #     self.epoch = epoch

    def __len__(self):
        if self.is_train:
            return len(self.dataset)
        else:
            return len(self.dataset)
            # return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int):
        # 'atoms', 'coordinates', 'smi', 'scaffold', 'target', 'ori_index'
        
        # sample multiple conformers
        
        
        atoms = np.array(self.dataset[index][self.atoms])
        assert len(atoms) > 0
        size = len(self.dataset[index][self.coordinates])
        sample_idx = np.random.randint(size)
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        
        
        # if self.is_train:
        #     atoms = np.array(self.dataset[index][self.atoms])
        #     assert len(atoms) > 0
        #     size = len(self.dataset[index][self.coordinates])
        #     sample_idx = np.random.randint(size)
        #     coordinates = self.dataset[index][self.coordinates][sample_idx]
        # else:
        #     smi_idx = index // self.conf_size
        #     coord_idx = index % self.conf_size
        #     atoms = np.array(self.dataset[smi_idx][self.atoms])
        #     coordinates = np.array(self.dataset[smi_idx][self.coordinates][coord_idx])
        #     index = smi_idx
        
        
        targets = self.dataset[index]['target']
        # org_coords, noisy_coords
        results = {"atoms": atoms, "coordinates": coordinates.astype(np.float32), "targets": targets}
        if 'mol' in self.dataset[index]:
            mol = self.dataset[index]['mol']
            results['mol'] = mol
            
            
        res_data = Data()
        res_data.smi = self.dataset[index]['smi']
        res_data.pos = torch.tensor(results['coordinates'], dtype=torch.float)
        # res_data.z = # TODO get number of atom type
        res_data.z = torch.tensor([period_atomic_number[ele] for ele in results["atoms"]], dtype=torch.long)
        
        if res_data.pos.shape[0] > self.max_size:
            print(f'cut off smi {res_data.smi} to max size {self.max_size}')
            indices = torch.randperm(res_data.pos.shape[0])[:self.max_size]
            res_data.pos = res_data.pos[indices]
            res_data.z = res_data.z[indices]

        
        res_data.y = torch.tensor(results['targets'])
        if res_data.y.shape[0] > 1: # multi task
            res_data.y = res_data.y.reshape(1, -1)
        
        if self.transform_noise is not None:
            res_data = self.transform_noise(res_data)
        
        
        return self.filter_max(res_data)
        
        # return res_data
    
    def filter_max(self, data):
        # filter pos, z, pos_target
        if data.pos.shape[1] > self.max_size:
            # random masking
            select_idx = torch.randprem(data.pos.shape[1])[:self.max_size]
            data.pos = data.pos[select_idx]
            data.z = data.z[select_idx]
            data.pos_target = data.pos_target[select_idx]
        return data

    def __getitem__(self, index: int):
        return self.__cached_item__(index)


DS_Tasks = ['ADRB2', 'ABL1', 'CYT2C9', 'PPARG', 'GluA2', '3CL', 'HIVINT', 'HDAC2', 'KRAS', 'PDE5']
sub_tasks = ['docking_score', 'emodel_score', 'hbond_score', 'polar_score', 'coulombic_score']

class DockingDataset(Dataset):
    def __init__(self,
                 data_path='/home/AI4Science/fengsk/DockingData/dataset_v2.csv', task='ADRB2', split='train', pkl_path='/home/AI4Science/fengsk/DockingData/diversity_molecule_set.pkl', idx_data_path='/home/AI4Science/fengsk/DockingData/docking_id_idx_map.json', transform=None):

        all_data = pd.read_csv(data_path)
        split_lst = ['train', 'valid', 'test']
        assert split in split_lst
        sidx = split_lst.index(split)
        self.train_data = all_data[all_data['scaffold_folds'] == sidx]

        assert task in DS_Tasks
        this_task = [f'{task}_{st}' for st in sub_tasks]

        self.id_lst = self.train_data['IDs'].tolist()
        labels = []
        for tk in this_task:
            labels.append(self.train_data[tk].tolist())

        labels = np.array(labels)
        self.labels = np.transpose(labels)
        self.length = len(self.id_lst)

        with open(pkl_path, 'rb') as pa:
            self.raw_datas = pickle.load(pa)

        with open(idx_data_path, 'r') as fp:
            self.idx_map = json.load(fp)
        
        self.transform_noise = transform



    def __len__(self):
        # test
        # return 100
        return len(self.labels)

    def __getitem__(self, index):
        id = self.id_lst[index]
        idx = self.idx_map[id]
        raw_data = self.raw_datas[idx]
        res_data = Data()
        res_data.y = torch.tensor([self.labels[index]], dtype=torch.float)
        res_data.pos = torch.tensor(raw_data['coordinates'][0], dtype=torch.float)
        
        # res_data.z = # TODO get number of atom type
        res_data.z = torch.tensor([period_atomic_number[ele] for ele in raw_data["atoms"]], dtype=torch.long)
        
        if self.transform_noise is not None:
            res_data = self.transform_noise(res_data)
        
        return res_data
        # Process your sample here if needed
        rdkit_mol = AllChem.MolFromSmiles(smile)
        data = mol_to_graph_data_obj_simple(rdkit_mol)
        # manually add mol id
        data.id = index
        data.y = torch.tensor(self.labels[index], dtype=torch.float32)        
        data.y = data.y.reshape(1, -1)
        return data    

# molnet test
# class TTADataset(InMemoryDataset):
    # def __init__(self, dataset_path, dataset_name, conf_size=10):
    #     self.dataset = LMDBDataset(dataset_path)
    #     self.dataset_name = dataset_name
    #     self.atoms = 'atoms'
    #     self.coordinates = 'coordinates'
    #     self.conf_size = conf_size
    #     self.set_epoch(None)

    # def set_epoch(self, epoch, **unused):
    #     super().set_epoch(epoch)
    #     self.epoch = epoch

    # def __len__(self):
    #     return len(self.dataset) * self.conf_size

    # @lru_cache(maxsize=16)
    # def __cached_item__(self, index: int, epoch: int):
    #     smi_idx = index // self.conf_size
    #     coord_idx = index % self.conf_size
    #     atoms = np.array(self.dataset[smi_idx][self.atoms])
    #     coordinates = np.array(self.dataset[smi_idx][self.coordinates][coord_idx])
    #     smi = self.dataset[smi_idx]["smi"]
    #     target = self.dataset[smi_idx]["target"]
    #     return {
    #         "atoms": atoms,
    #         "coordinates": coordinates.astype(np.float32),
    #         "smi": smi,
    #         "target": target,
    #     }

    # def __getitem__(self, index: int):
    #     return self.__cached_item__(index, self.epoch)