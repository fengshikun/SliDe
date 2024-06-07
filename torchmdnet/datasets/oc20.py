import torch
from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict
import os
import numpy as np
from torch.utils.data import Dataset
import lmdb
from typing import Any, Callable, List, Optional, Tuple, Union
from collections.abc import Sequence
from torch import Tensor
IndexType = Union[slice, Tensor, np.ndarray, Sequence]
from rdkit import Chem
import pickle as pkl

from torsion_utils import get_torsions, GetDihedral, apply_changes, add_equi_noise_new
import copy
from torch_geometric.data import (InMemoryDataset, Data)

class oc20_is2re(InMemoryDataset):
    def __init__(self, data_path, transform_noise=None, lp_sep=False):

        env = lmdb.open(data_path, subdir=False)
        self.txn = env.begin(write=False)
        self.transform_noise = transform_noise
        self.length = int(self.txn.stat()['entries'])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):

        indata = pkl.loads(self.txn.get(str(idx).encode()))
        indata = Data(**indata.__dict__) # versioin problem

        data = Data()
        data.z = torch.as_tensor(indata.atomic_numbers, dtype=torch.long)
        data.pos = torch.as_tensor(indata.pos, dtype=torch.float32)
        if self.transform_noise is not None:
            data = self.transform_noise(data) # noisy node
        data.y = torch.as_tensor(indata.y_relaxed, dtype=torch.float32)
        data.pos_relaxed = torch.as_tensor(indata.pos_relaxed, dtype=torch.float32)

        return data