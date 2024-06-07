from torch_geometric.data import Data, Dataset
import numpy as np
import torch
from torch_geometric.data import (InMemoryDataset, Data)
import mmap
import pickle as pk


atomic_number = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
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
                 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118, 'D': 119}

# atomic_number = ['C', 'O', 'S', 'N', 'H', 'I', 'Br', 'Cl', 'F', 'P', 'He', 'Li', 'Be', 'B', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'D']

class Pocket(InMemoryDataset):
    def __init__(self, pocket_data, idx_path, transform_noise=None):
        # get file handler
        pocket_r = open(pocket_data,'r+b')
        self.pocket_handler = mmap.mmap(pocket_r.fileno(), 0)
        # read index
        with open(idx_path, 'r') as ir:
            idx_info_lst = ir.readlines()
        
        self.idx_lst = []
        for ele in idx_info_lst:
            ele_array = [int(i) for i in ele.strip().split()]
            self.idx_lst.append(ele_array)
        self.length = len(self.idx_lst)
        self.transform_noise = transform_noise
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx, drop_atom_lst=['H', 'D']):

        data = Data()
        # get z, pos, and y
        # read element
        idx_array = self.idx_lst[idx]
        org_data = pk.loads(self.pocket_handler[idx_array[0]: idx_array[0] + idx_array[1]])
        pocket_atoms = org_data['pocket_atoms']
        pocket_coordinates = org_data['pocket_coordinates']
        # pocket_z = [atomic_number.index(ele) for ele in pocket_atoms]
        pocket_z = [atomic_number[ele] for ele in pocket_atoms]
        
        lig_atoms_real = org_data['lig_atoms_real']
        lig_coord_real = org_data['lig_coord_real']
        # lig_z = [atomic_number.index(ele) for ele in lig_atoms_real]
        lig_z = [atomic_number[ele] for ele in lig_atoms_real]


        num_atoms = len(pocket_atoms) + len(lig_atoms_real)

        # concat z and pos
        pocket_atoms.extend(lig_atoms_real)
        pocket_z.extend(lig_z)
        all_pos = np.concatenate([pocket_coordinates, lig_coord_real])

        poc_lig_id = np.zeros(num_atoms)
        poc_lig_id[len(pocket_atoms): ] = 1 # lig 1

        data.z = torch.tensor(pocket_z, dtype=torch.long) 
        data.pos = torch.tensor(all_pos, dtype=torch.float32)
        data.type_mask = torch.tensor(poc_lig_id, dtype=torch.long)

        if len(drop_atom_lst): # erase H
            pocket_atoms = np.array(pocket_atoms)
            mask_idx = np.in1d(pocket_atoms, drop_atom_lst)
            data.z = data.z[~mask_idx]
            data.pos = data.pos[~mask_idx]
            data.type_mask = data.type_mask[~mask_idx]

        
        

        if self.transform_noise is not None:
            data = self.transform_noise(data) # noisy node
        
        return data


import matplotlib.pyplot as plt
from tqdm import tqdm
def draw_dist(dist_numpy, save_prefix='', delima=0.01):
    plt.cla()
    bins = np.arange(dist_numpy.min(), dist_numpy.max(), delima)
    plt.hist(dist_numpy, bins=bins, label='pos', color='steelblue', alpha=0.7)
    plt.title(f'{save_prefix}')
    plt.xlabel('value')
    plt.ylabel('number')
    plt.savefig(f'{save_prefix}.png')

if __name__=='__main__':
    # x = open('/home/fengshikun/DenoisingData/prot_frag/ligand_pocket_samples.data','r+b')
    # y = mmap.mmap(x.fileno(), 0)
    # z = pk.loads(y[0:6224])
    # print(z)
    pocket_data = '/data/protein/SKData/PocData/new_data/ligand_pocket_new.data'
    pocket_idx = '/data/protein/SKData/PocData/new_data/ligand_pocket_new.index'

    atom_type = ['C', 'O', 'S', 'N', 'H', 'I', 'Br', 'Cl', 'F', 'P']

    for atom in atomic_number:
        if atom not in atom_type:
            atom_type.append(atom)

    print(atom_type)

    # p_data = Pocket(pocket_data, pocket_idx)
    # atom_num_lst = []
    # fail_lst = []
    # p_len = len(p_data)
    # for i in tqdm(range(p_len)):
    #     try:
    #         data = p_data[i]
    #     except Exception as e:
    #         print(f'idx {i} fail with exceptioin {e}')
    #         fail_lst.append(i)
    #         continue
    #     atom_num = data.z.shape[0]
    #     atom_num_lst.append(atom_num)
    # np.save('pocket_atom_num.npy', atom_num_lst)
    # atom_num_lst = np.load('pocket_atom_num.npy')
    # draw_dist(atom_num_lst, 'pocket_data_num', delima=1)