from typing import Optional, Callable, List

import os
from tqdm import tqdm
import glob
import ase
import numpy as np
from rdkit import Chem
from torchmdnet.utils import isRingAromatic, get_geometry_graph_ring
from typing import Any, Callable, List, Optional, Tuple, Union
from collections.abc import Sequence
from torch import Tensor
IndexType = Union[slice, Tensor, np.ndarray, Sequence]
import random
import torch.nn.functional as F
import copy
import lmdb
import pickle

import torch
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)

from torsion_utils import get_torsions, GetDihedral, apply_changes, get_rotate_order_info, add_equi_noise, get_2d_gem, get_info_by_gem_idx, add_equi_keep_noise, transform_noise, check_in_samering, GetBondLength, SetBondLength, GetAngle, SetAngle, GetDihedral, SetDihedral, concat_idx_label
import math
from rdkit.Geometry import Point3D



def add_equi_noise_new(opt_mol, bond_var=0.04, angle_var=0.143, torsion_var_r=2.8, torsion_var=0.41, coord_var=0.04, add_ring_noise=False, add_noise_type=0, mol_param=None, ky=0):
    # bond noise, find all bond add noise
    
    org_conf = opt_mol.GetConformer()
    mol = copy.deepcopy(opt_mol)
    conf = mol.GetConformer()
    
    atom_num = mol.GetNumAtoms()
    pos_coords = np.zeros((atom_num, 3), dtype=np.float32)
        # pos_noise_coords = new_mol.GetConformer().GetPositions()
    for idx in range(atom_num):
        c_pos = conf.GetAtomPosition(idx)
        pos_coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
    
    
    if add_noise_type == 2: # gaussain noise
        data_noise = transform_noise(pos_coords)
        # set the data_noise back to the mol
        for i in range(atom_num):
            x,y,z = data_noise[i]
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        rotate_dihedral_label_lst = []
    elif add_noise_type == 1: # Frad noise
        rotable_bonds = get_torsions([mol])
        org_angle = []
        for rot_bond in rotable_bonds:
            org_angle.append(GetDihedral(mol.GetConformer(), rot_bond))
        org_angle = np.array(org_angle)        
        noise_angle = transform_noise(org_angle, position_noise_scale=2)
        new_mol = apply_changes(mol, noise_angle, rotable_bonds) # add the 
        coord_conf = new_mol.GetConformer()
        
        pos_noise_coords_angle = np.zeros((atom_num, 3), dtype=np.float32)
        # pos_noise_coords = new_mol.GetConformer().GetPositions()
        for idx in range(atom_num):
            c_pos = coord_conf.GetAtomPosition(idx)
            pos_noise_coords_angle[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
        # add guassian noise
        pos_noise_coords = transform_noise(pos_noise_coords_angle, position_noise_scale=0.04)
        
        # set back to the mol
        opt_mol = copy.deepcopy(new_mol) # clone first
        for i in range(atom_num):
            x,y,z = pos_noise_coords[i]
            coord_conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol = new_mol
        
        rotate_dihedral_label_lst = []
    elif add_noise_type == 0:
    
        if add_ring_noise:
            ring_mask = []
            for atom in mol.GetAtoms(): # j
                if atom.IsInRing():
                    ring_mask.append(True)
                else:
                    ring_mask.append(False)
            noise = torch.randn_like(torch.tensor(pos_coords)) * coord_var
            data_noise = pos_coords + noise.numpy()
            # set back to conf
            for i in range(atom_num):
                if ring_mask[i]: # only add ring noise
                    x,y,z = data_noise[i]
                    conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
            

        # get ring info
        ring_array = []
        for ring in mol.GetRingInfo().AtomRings():
            ring_array.append(ring)

        set_bond_idx = []
        for bond in mol.GetBonds():
            i_idx = bond.GetBeginAtomIdx()
            j_idx = bond.GetEndAtomIdx()
            # if mol.GetAtomWithIdx(i_idx).IsInRing() and mol.GetAtomWithIdx(j_idx).IsInRing():
            #     continue
            if check_in_samering(i_idx, j_idx, ring_array):
                continue

            org_bond_len = GetBondLength(conf, [i_idx, j_idx])
            
            if mol_param is not None:
                try:
                    bond_para = mol_param[0]["Bonds"][(i_idx, j_idx)].k._magnitude
                    bond_item_var = np.sqrt(1 / bond_para)
                except Exception as e:
                    # print(f'Exception {e}, No bond param for key {ky}, bond: {i_idx}_{j_idx}')
                    bond_item_var = bond_var
            else:
                bond_item_var = bond_var
            # add gaussian noise:
            noise_bond_len = np.random.normal(loc=org_bond_len, scale=bond_item_var)
            # set bond_length
            SetBondLength(conf, [i_idx, j_idx], noise_bond_len)
            set_bond_idx.append([i_idx, j_idx])
        # angle noise
        set_angle_idx = []
        for atom in mol.GetAtoms(): # j
            j_idx = atom.GetIdx()
            # atom_symbol = atom.GetSymbol()
            atom_degree = atom.GetDegree()
            if atom_degree >= 2:
                # get neighbors
                neighbors = atom.GetNeighbors()
                neb_lst = []
                for neb in neighbors:
                    neb_lst.append(neb.GetIdx())

                if mol.GetAtomWithIdx(j_idx).IsInRing(): # if j in ring, must pick one neb in ring as i
                    for n_idx in neb_lst:
                        if check_in_samering(j_idx, n_idx, ring_array):
                            i_idx = n_idx
                            break
                else:
                    # j not in ring, random pick one as i
                    i_idx = random.choice(neb_lst)
                
                neb_lst.remove(i_idx)
                # iterate k
                for k_idx in neb_lst:
                    # judge (i, j) and (j, k) in ring:
                    if check_in_samering(i_idx, j_idx, ring_array) and check_in_samering(j_idx, k_idx, ring_array):
                        continue
                    # add angle noise to (i, j, k)
                    org_angle = GetAngle(conf, [i_idx, j_idx, k_idx])
                    if math.isnan(org_angle): # may be nan
                        continue
                    # add noise
                    if mol_param is not None:
                        try:
                            angle_para = mol_param[0]["Angles"][(i_idx, j_idx, k_idx)].k._magnitude
                            angle_item_var = np.sqrt(1 / angle_para)
                        except Exception as e:
                            # print(f'Exception {e}, No angle param for key {ky}, angle: {i_idx}_{j_idx}_{k_idx}')
                            angle_item_var = angle_var
                    else:
                        angle_item_var = angle_var
                    # FIX by change the unit to radian from degree
                    noise = np.random.normal(scale=angle_item_var)
                    noise_angle = org_angle + noise # np.rad2deg(noise)
                    # * 57.3
                    if noise_angle >= 180:
                        noise_angle = 360 - noise_angle # cut the influence to the dih
                    elif noise_angle <= 0:
                        continue
                    
                    # valid_value = wiki_dihedral(conf, [14, 0, 3, 5])
                    # print(f"before add angel noise, [14, 0, 3, 5] dihedral is {valid_value}")
                    
                    # noise_angle = np.random.normal(loc=org_angle, scale=angle_var)
                    SetAngle(conf, [i_idx, j_idx, k_idx], noise_angle)
                    
                    set_angle_idx.append([i_idx, j_idx, k_idx])
                    # valid_value = wiki_dihedral(conf, [14, 0, 3, 5])
                    # print(f"after add angel noise, [14, 0, 3, 5] dihedral is {valid_value}, noise is {noise}")
                    # if valid_value > 175:
                    #     print('debug')
        
        # dihedral angle(rotatable or not) [i, j, k, l]
        # get the all the rotatable angel idx
        rotable_bonds = get_torsions([mol]) # format like [(0, 5, 10, 7), (1, 6, 12, 11), (6, 12, 11, 4)]
        
        rotable_sets = set([])
        for rb in rotable_bonds:
            rotable_sets.add(f'{rb[1]}_{rb[2]}')
            rotable_sets.add(f'{rb[2]}_{rb[1]}')

        # dihedral_label_lst = [] # [i, j, k, l, delta_angle]
        set_torsion_idx = []
        rotate_dihedral_label_lst = []
        for bond in mol.GetBonds():

            is_rotate = False

            j_idx = bond.GetBeginAtomIdx()
            k_idx = bond.GetEndAtomIdx()
            # check (j_idx, k_idx) in ring or not
            if check_in_samering(j_idx, k_idx, ring_array):
                continue

            
            j_atom = mol.GetAtomWithIdx(j_idx)
            j_atom_degree = j_atom.GetDegree()
            k_atom = mol.GetAtomWithIdx(k_idx)
            k_atom_degree = k_atom.GetDegree()

            if j_atom_degree < 2 or k_atom_degree < 2: # cannot compose a dihedral angle
                continue

            # get neibors
            j_neighbors = j_atom.GetNeighbors()
            j_neb_lst = []
            for neb in j_neighbors:
                j_neb_lst.append(neb.GetIdx())
            j_neb_lst.remove(k_idx)

            k_neighbors = k_atom.GetNeighbors()
            k_neb_lst = []
            for neb in k_neighbors:
                k_neb_lst.append(neb.GetIdx())
            k_neb_lst.remove(j_idx)

            # random pick one neighbor from j and k, taken as i, l
            i_idx = random.choice(j_neb_lst)
            l_idx = random.choice(k_neb_lst)

            if f'{j_idx}_{k_idx}' in rotable_sets: # rotatable
                deh_var = torsion_var_r
                is_rotate = True
            else:
                deh_var = torsion_var
                is_rotate = False
            
            if mol_param is not None:
                try:
                    torsion_para = mol_param[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].k[0]._magnitude 
                    torsion_period = mol_param[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].periodicity[0]  #for torsion i-j-k-l
                    
                    if torsion_para == 0:
                        continue
                    
                    sqrt_value = 1 / (torsion_para*(torsion_period**2))
                    if sqrt_value < 0:
                        continue
                                      
                    deh_var_item = np.sqrt(sqrt_value)
                except Exception as e:
                    deh_var_item = deh_var
                    # print(f'Exception {e}, No torsion param for key {ky}, torsion: {i_idx}_{j_idx}_{k_idx}_{l_idx}')
                    
                
            else:
                deh_var_item = deh_var
            # torsion_para = para[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].k[0]._value 
# torsion_period=para[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].periodicity[0]  #for torsion i-j-k-l
            
            
            # FIX by change the unit to radian from degree
            noise = np.random.normal(scale=deh_var_item)
            

            # noise_deh_angle = np.random.normal(loc=org_deh_angle, scale=deh_var)
            for l_idx in k_neb_lst:
                # add noise
                org_deh_angle = GetDihedral(conf, [i_idx, j_idx, k_idx, l_idx])
                if math.isnan(org_deh_angle): # may be nan
                    continue
                noise_deh_angle = org_deh_angle + noise # np.rad2deg(noise)
                # valid_value = wiki_dihedral(conf, [14, 0, 3, 5])
                # print(f"before add noise, [14, 0, 3, 5] dihedral is {valid_value}")
                SetDihedral(conf, [i_idx, j_idx, k_idx, l_idx], noise_deh_angle)
                
                set_torsion_idx.append([i_idx, j_idx, k_idx, l_idx])
                # valid_value = wiki_dihedral(conf, [14, 0, 3, 5])
                # print(f"after add noise, [14, 0, 3, 5] dihedral is {valid_value}, noise is {noise}")
                # if valid_value > 175:
                #     print('debug')

        #     if is_rotate:
        #         rotate_dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_deh_angle - org_deh_angle])
        #     else:
        #         dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_deh_angle - org_deh_angle])
        
        # get the difference between mol and opt_mol
    # edge_idx, angle_idx, dihedral_idx = get_2d_gem(opt_mol)
    
    edge_idx, angle_idx, dihedral_idx = set_bond_idx, set_angle_idx, set_torsion_idx

    bond_len_lst, angle_lst, dihedral_lst = get_info_by_gem_idx(opt_mol, edge_idx, angle_idx, dihedral_idx)
    bond_len_lst_noise, angle_lst_noise, dihedral_lst_noise = get_info_by_gem_idx(mol, edge_idx, angle_idx, dihedral_idx)
    # noise difference, has filter nan
    bond_label_lst, angle_label_lst, dihedral_label_lst = concat_idx_label(edge_idx, angle_idx, dihedral_idx, bond_len_lst_noise-bond_len_lst, angle_lst_noise-angle_lst, dihedral_lst_noise-dihedral_lst)
    # edge: [i,j, noise(i,j)]
    # angle: [i, j, k, noise(i,j,k)]
    # filter zero
    if angle_label_lst.size != 0:
        angle_label_lst = angle_label_lst[angle_label_lst[:,-1] != 0]

    # filter zero
    if dihedral_label_lst.size != 0:
        dihedral_label_lst = dihedral_label_lst[dihedral_label_lst[:, -1] != 0]

    if add_noise_type == 1:
        mol = [mol, opt_mol]
    
    
    if len(bond_label_lst): # denan
        bond_label_lst = np.array(bond_label_lst, dtype=np.float32)
        # bond_label_lst[:,2] = noise_bond - org_bond
        mask = ~np.isnan(bond_label_lst[:, 2])
        bond_label_lst = bond_label_lst[mask]
        
    if len(angle_label_lst): # denan
        angle_label_lst = np.array(angle_label_lst, dtype=np.float32)
        # angle_label_lst[:, 3] = noise_angle - org_angle
        mask = ~np.isnan(angle_label_lst[:, 3])
        angle_label_lst = angle_label_lst[mask]
        
    
    specific_var_lst = []
    if mol_param is not None:
        for bond_label in bond_label_lst:
            i_idx, j_idx = int(bond_label[0]), int(bond_label[1])
            try:
                bond_para = mol_param[0]["Bonds"][(i_idx, j_idx)].k._magnitude
                bond_item_var = np.sqrt(1 / bond_para)
            except Exception as e:
                # print(f'Exception {e}, No bond param for key {ky}, bond: {i_idx}_{j_idx}')
                bond_item_var = bond_var
            specific_var_lst.append(bond_item_var)
        for angle_label in angle_label_lst:
            i_idx, j_idx, k_idx = int(angle_label[0]), int(angle_label[1]), int(angle_label[2])
            try:
                angle_para = mol_param[0]["Angles"][(i_idx, j_idx, k_idx)].k._magnitude
                angle_item_var = np.sqrt(1 / angle_para)
            except Exception as e:
                # print(f'Exception {e}, No angle param for key {ky}, angle: {i_idx}_{j_idx}_{k_idx}')
                angle_item_var = angle_var
            specific_var_lst.append(angle_item_var)
        
        for torsion_label in dihedral_label_lst:
            i_idx, j_idx, k_idx, l_idx = int(torsion_label[0]), int(torsion_label[1]), int(torsion_label[2]), int(torsion_label[3]),
            
            if f'{j_idx}_{k_idx}' in rotable_sets: # rotatable
                deh_var = torsion_var_r
                is_rotate = True
            else:
                deh_var = torsion_var
                is_rotate = False
            
            try:
                torsion_para = mol_param[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].k[0]._magnitude 
                torsion_period = mol_param[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].periodicity[0]
                if torsion_para == 0:
                    deh_var_item = deh_var
                else:
                    sqrt_value = 1 / (torsion_para*(torsion_period**2))
                    if sqrt_value < 0:
                        deh_var_item = deh_var
                    else:                
                        deh_var_item = np.sqrt(sqrt_value)
            except Exception as e:
                deh_var_item = deh_var
            
            specific_var_lst.append(deh_var_item)
    
    
    
    return mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst, specific_var_lst
    # return mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst



class PCQM4MV2_XYZ(InMemoryDataset):
    r"""3D coordinates for molecules in the PCQM4Mv2 dataset (from zip).
    """

    raw_url = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2_xyz.zip'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None):
        assert dataset_arg is None, "PCQM4MV2 does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['pcqm4m-v2_xyz']

    @property
    def processed_file_names(self) -> str:
        return 'pcqm4mv2__xyz.pt'

    def download(self):
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

    def process(self):
        dataset = PCQM4MV2_3D(self.raw_paths[0])
        
        data_list = []
        for i, mol in enumerate(tqdm(dataset)):
            pos = mol['coords']
            pos = torch.tensor(pos, dtype=torch.float)
            z = torch.tensor(mol['atom_type'], dtype=torch.long)

            data = Data(z=z, pos=pos, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


# Globle variable
MOL_LST = None
MOL_DEBUG_LST = None
debug = False
debug_cnt = 0
class PCQM4MV2_XYZ_BIAS(PCQM4MV2_XYZ):
    #  sdf path: pcqm4m-v2-train.sdf
    # set the transform to None
    def __init__(self, root: str, sdf_path: str, position_noise_scale: float, sample_number: int, violate: bool, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None):
        assert dataset_arg is None, "PCQM4MV2_XYZ_BIAS does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.suppl = Chem.SDMolSupplier(sdf_path)
        self.position_noise_scale = position_noise_scale
        self.sample_number = sample_number
        self.violate = violate
        global MOL_LST
        if MOL_LST is None:
            import pickle
            with open(sdf_path, 'rb') as handle:
                MOL_LST = pickle.load(handle)
        if debug:
            global MOL_DEBUG_LST
            if MOL_DEBUG_LST is None:
                # MOL_DEBUG_LST = Chem.SDMolSupplier("pcqm4m-v2-train.sdf")
                MOL_DEBUG_LST = np.load("mol_iter_all.npy", allow_pickle=True)
        # import pickle
        # with open(sdf_path, 'rb') as handle:
        #     self.mol_lst = pickle.load(handle)

        print('PCQM4MV2_XYZ_BIAS Initialization finished')

    def transform_noise(self, data, position_noise_scale):
        noise = torch.randn_like(data) * position_noise_scale
        data_noise = data + noise
        return data_noise

    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', Data]:
        org_data = super().__getitem__(idx)
        # change org_data coordinate
        # get mol
        molinfo = MOL_LST[idx.item()]
        edges_src, edges_dst, org_coordinate = molinfo
        atom_woh_number = org_coordinate.shape[0]
        
        coords = org_data.pos

        repeat_coords = coords.unsqueeze(0).repeat(self.sample_number, 1, 1)
        noise_coords = self.transform_noise(repeat_coords, self.position_noise_scale)
        noise_feat = torch.linalg.norm(noise_coords[:,edges_src] - noise_coords[:,edges_dst], dim=2)
        feat = torch.linalg.norm(coords[edges_src] - coords[edges_dst], dim=1)
        loss_lst = torch.mean((noise_feat**2 - feat ** 2)**2, dim=1)
        # sorted_value, sorted_idx = torch.sort(loss_lst)
        
        # min_violate_idx, max_violate_idx = sorted_idx[0], sorted_idx[-1]
        
        if self.violate:
            # new_coords = noise_coords[max_violate_idx]
            new_coords = noise_coords[torch.argmax(loss_lst)]
        else:
            # new_coords = noise_coords[min_violate_idx]
            new_coords = noise_coords[torch.argmin(loss_lst)]
        
        org_data.pos_target = new_coords - coords
        org_data.pos = new_coords
        
        global debug_cnt
        if debug:
            import copy
            from rdkit.Geometry import Point3D
            mol = MOL_DEBUG_LST[idx.item()]
            violate_coords = noise_coords[torch.argmax(loss_lst)].cpu().numpy()
            n_violate_coords = noise_coords[torch.argmin(loss_lst)].cpu().numpy()
            mol_cpy = copy.copy(mol)
            conf = mol_cpy.GetConformer()
            for i in range(mol.GetNumAtoms()):
                x,y,z = n_violate_coords[i]
                conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
            

            writer = Chem.SDWriter(f'org_{debug_cnt}.sdf')
            writer.write(mol)
            writer.close()

            # supplier = Chem.SDMolSupplier('v3000.sdf')
            writer = Chem.SDWriter(f'min_noise_{debug_cnt}.sdf')
            writer.write(mol_cpy)
            writer.close()
            # show mol coordinate
            mol_cpy = copy.copy(mol)
            conf = mol_cpy.GetConformer()
            for i in range(mol.GetNumAtoms()):
                x,y,z = violate_coords[i]
                conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

            writer = Chem.SDWriter(f'max_noise_{debug_cnt}.sdf')
            writer.write(mol_cpy)
            writer.close()
            debug_cnt += 1
            if debug_cnt > 10:
                exit(0)

        return org_data


class PCQM4MV2_Dihedral(PCQM4MV2_XYZ):
    def __init__(self, root: str, sdf_path: str, dihedral_angle_noise_scale: float, position_noise_scale: float, composition: bool, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None):
        assert dataset_arg is None, "PCQM4MV2_Dihedral does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.suppl = Chem.SDMolSupplier(sdf_path)
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.position_noise_scale = position_noise_scale
        self.composition = composition # compose dihedral angle and position noise
        global MOL_LST
        if MOL_LST is None:
            # import pickle
            # with open(sdf_path, 'rb') as handle:
            #     MOL_LST = pickle.load(handle)
            # MOL_LST = np.load("mol_iter_all.npy", allow_pickle=True)
                MOL_LST = np.load("h_mol_lst.npy", allow_pickle=True)

        if debug:
            global MOL_DEBUG_LST
            if MOL_DEBUG_LST is None:
                # MOL_DEBUG_LST = Chem.SDMolSupplier("pcqm4m-v2-train.sdf")
                MOL_DEBUG_LST = np.load("mol_iter_all.npy", allow_pickle=True)
    
    def transform_noise(self, data, position_noise_scale):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
        data_noise = data + noise.numpy()
        return data_noise

    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', Data]:
        org_data = super().__getitem__(idx)
        org_atom_num = org_data.pos.shape[0]
        # change org_data coordinate
        # get mol
        mol = MOL_LST[idx.item()]
        atom_num = mol.GetNumAtoms()
        if atom_num != org_atom_num:
            pos_noise_coords = self.transform_noise(org_data.pos, self.position_noise_scale)
            org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos)
            org_data.pos = torch.tensor(pos_noise_coords)
        
            return org_data


        coords = np.zeros((atom_num, 3), dtype=np.float32)
        coord_conf = mol.GetConformer()
        for idx in range(atom_num):
            c_pos = coord_conf.GetAtomPosition(idx)
            coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
        # coords = mol.GetConformer().GetPositions()

        
        # get rotate bond
        rotable_bonds = get_torsions([mol])
        
        if self.composition or not len(rotable_bonds):
            pos_noise_coords = self.transform_noise(coords, self.position_noise_scale)
            if len(rotable_bonds): # set coords into the mol
                conf = mol.GetConformer()
                for i in range(mol.GetNumAtoms()):
                    x,y,z = pos_noise_coords[i]
                    conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        

        if len(rotable_bonds):
            org_angle = []
            for rot_bond in rotable_bonds:
                org_angle.append(GetDihedral(mol.GetConformer(), rot_bond))
            org_angle = np.array(org_angle)        
            noise_angle = self.transform_noise(org_angle, self.dihedral_angle_noise_scale)
            new_mol = apply_changes(mol, noise_angle, rotable_bonds)
            
            coord_conf = new_mol.GetConformer()
            pos_noise_coords = np.zeros((atom_num, 3), dtype=np.float32)
            # pos_noise_coords = new_mol.GetConformer().GetPositions()
            for idx in range(atom_num):
                c_pos = coord_conf.GetAtomPosition(idx)
                pos_noise_coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]

        
        org_data.pos_target = torch.tensor(pos_noise_coords - coords)
        org_data.pos = torch.tensor(pos_noise_coords)
        
        return org_data

# target is the noise self.
def handle_abnormal_torsioin(target):
    less_neg_180 = (target < -180)
    target[less_neg_180] += 360
    # target[less_neg_180] = -target[less_neg_180]
    
    big_than_180 = (target > 180)
    target[big_than_180] -= 360
    # target[big_than_180] = -target[big_than_180]
    return target
    


# use force filed definition
# bond length, angle ,dihedral angel

Param_Lst = None

class PCQM4MV2_Force(PCQM4MV2_XYZ):
    def __init__(self, root: str, sdf_path: str, dihedral_angle_noise_scale: float, angle_noise_scale: float, nt_noise_scale: float, bond_length_scale: float, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None, add_ring_noise=False, derivative_var=0, sample_num=32, add_noise_type=0, use_force_param=0, dev_simga=0.001, ring_coord_var=0.04, frad_guassian_scale=0.0):
        assert dataset_arg is None, "PCQM4MV2_Dihedral does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.suppl = Chem.SDMolSupplier(sdf_path)
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale # rotate torsion
        self.nt_noise_scale = nt_noise_scale # non rotate torsion
        self.angle_noise_scale = angle_noise_scale # angle 
        self.bond_length_scale = bond_length_scale # bond
        self.ring_coord_var = ring_coord_var
        
        
        
        self.add_ring_noise = add_ring_noise
        
        self.derivative_var = derivative_var
        self.dev_sigma = dev_simga
        self.sample_num = sample_num
        
        self.add_noise_type = add_noise_type # add noise type: default: 0 BAT noise, 1 Frad noise, 2 coordinate guassian noise;
        global MOL_LST
        self.use_force_param = use_force_param
        
        
        self.frad_guassian_scale = frad_guassian_scale # frad guassian scale 0.04
        
        if MOL_LST is None:
            # import pickle
            # with open(sdf_path, 'rb') as handle:
            #     MOL_LST = pickle.load(handle)
            # MOL_LST = np.load("mol_iter_all.npy", allow_pickle=True)
            # MOL_LST = np.load("h_mol_lst.npy", allow_pickle=True)
            MOL_LST = lmdb.open('/data/protein/SKData/DenoisingData/MOL_LMDB', readonly=True, subdir=True, lock=False)
        
        global Param_Lst
        if Param_Lst is None and self.use_force_param:
            Param_Lst = lmdb.open('/data/protein/SKData/DenoisingData/Param_DB', readonly=True, subdir=True, lock=False)
        
        if debug:
            global MOL_DEBUG_LST
            if MOL_DEBUG_LST is None:
                # MOL_DEBUG_LST = Chem.SDMolSupplier("pcqm4m-v2-train.sdf")
                MOL_DEBUG_LST = np.load("mol_iter_all.npy", allow_pickle=True)
    
    def transform_noise(self, data, position_noise_scale):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
        data_noise = data + noise.numpy()
        return data_noise

    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', Data]:
        org_data = super().__getitem__(idx)
        org_atom_num = org_data.pos.shape[0]
        # change org_data coordinate
        # get mol
        # mol = MOL_LST[idx.item()]

        ky = str(idx.item()).encode()
        serialized_data = MOL_LST.begin().get(ky)
        mol = pickle.loads(serialized_data)
        
        if self.use_force_param:
            serialized_data = Param_Lst.begin().get(ky)
            if serialized_data is None:
                print(f'key {ky}')
            mol_param = pickle.loads(serialized_data)
        else:
            mol_param = None
        
        org_data.pos_base = org_data.pos # NOTE add for esitimate FF
        org_data.eq_mol = mol   # NOTE add for esitimate FF
        # add noise to mol with different types of noise
        # noise_mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst = add_equi_noise(mol, bond_var=self.bond_length_scale, angle_var=self.angle_noise_scale, torsion_var=self.dihedral_angle_noise_scale, add_ring_noise = self.add_ring_noise)


        # noise_mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst = add_equi_noise_new(mol, bond_var=self.bond_length_scale, angle_var=self.angle_noise_scale, torsion_var=self.dihedral_angle_noise_scale, add_ring_noise = self.add_ring_noise)


        noise_mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst, bat_var_lst = add_equi_noise_new(mol, add_ring_noise=self.add_ring_noise, bond_var=self.bond_length_scale, angle_var=self.angle_noise_scale, torsion_var_r=self.dihedral_angle_noise_scale, torsion_var=self.nt_noise_scale, coord_var=self.ring_coord_var, add_noise_type=self.add_noise_type, mol_param=mol_param, ky=ky) # torsion_var_r=0.001, torsion_var=0.001


        if self.add_noise_type == 1:
            noise_mol, mol = noise_mol

        # get noise_mol coordinate
        atom_num = mol.GetNumAtoms()
        # assert atom_num == org_atom_num # todo, we may need handle such situation

        if atom_num != org_atom_num:
            print('assert atom_num == org_atom_num failed')
            atoms = mol.GetAtoms()
            z_lst = []
            for i in range(atom_num):
                atom = atoms[i]
                z_lst.append(atom.GetAtomicNum()) # atomic num start from 1
            
            org_data.z = torch.tensor(z_lst) # atomic num start from 1


        # get coordinates
        coords = np.zeros((atom_num, 3), dtype=np.float32)
        org_coords = np.zeros((atom_num, 3), dtype=np.float32)
        coord_conf = noise_mol.GetConformer()
        org_conf = mol.GetConformer()
        for idx in range(atom_num):
            c_pos = coord_conf.GetAtomPosition(idx)
            coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
            n_pos = org_conf.GetAtomPosition(idx)
            org_coords[idx] = [float(n_pos.x), float(n_pos.y), float(n_pos.z)]
        # coords = mol.GetConformer().GetPositions()
        

        
        
        # non_rotate_dihedral_len = len(dihedral_label_lst)
        # rotate_dihedral_len = len(rotate_dihedral_label_lst)
        
        # dihedral_all_label_lst = copy.copy(dihedral_label_lst)
        # dihedral_all_label_lst.extend(rotate_dihedral_label_lst) # all

        # org_bond, org_angle, org_dihedral = get_info_by_gem_idx(mol, edge_idx, angle_idx, dihedral_idx)
        # noise_bond, noise_angle, noise_dihedarl = get_info_by_gem_idx(noise_mol, edge_idx, angle_idx, dihedral_idx)


            
            
        
        if self.derivative_var:
            # fix bond, angle, dihedral label lst
            if len(bond_label_lst):
                edge_idx = np.array(bond_label_lst)[:,:2].astype(int).tolist()
            else:
                edge_idx = bond_label_lst # empty

            if len(angle_label_lst):
                angle_idx = np.array(angle_label_lst)[:,:3].astype(int).tolist()
            else:
                angle_idx = angle_label_lst # empty
            
            if len(dihedral_label_lst):
                dihedral_idx = np.array(dihedral_label_lst)[:,:4].astype(int).tolist()
            else:
                dihedral_idx = dihedral_label_lst # empty
            
            
            bond_len_lst_org, angle_lst_org, dihedral_lst_org = get_info_by_gem_idx(noise_mol, edge_idx, angle_idx, dihedral_idx)
            
            
            
            
            # confirm the variance
            rotable_bonds = get_torsions([mol]) # format like [(0, 5, 10, 7), (1, 6, 12, 11), (6, 12, 11, 4)]
        
            rotable_sets = set([])
            for rb in rotable_bonds:
                rotable_sets.add(f'{rb[1]}_{rb[2]}')
                rotable_sets.add(f'{rb[2]}_{rb[1]}')
            
            rotate_var_item = []
            for rv in dihedral_idx:
                j_idx, k_idx = rv[1], rv[2]
                if f'{j_idx}_{k_idx}' in rotable_sets: # rotatable
                    deh_var = self.dihedral_angle_noise_scale
                    is_rotate = True
                else:
                    deh_var = self.nt_noise_scale
                    is_rotate = False
                rotate_var_item.append(deh_var)
            
            bond_var_item = np.array([self.bond_length_scale for _ in range(len(edge_idx))])
            angle_var_item = np.array([self.angle_noise_scale for _ in range(len(angle_idx))])
            rotate_var_item = np.array(rotate_var_item)
            
            var_lst = np.concatenate((bond_var_item, angle_var_item, rotate_var_item))
            
            if self.use_force_param:
                var_lst = np.array(bat_var_lst)
            
            org_rlst = np.concatenate((bond_len_lst_org, angle_lst_org, dihedral_lst_org))
            
            
            
            if len(bond_label_lst):
                bond_delta = bond_label_lst[:,-1]
            else:
                bond_delta = []
            if len(angle_label_lst):
                angle_delta = angle_label_lst[:,-1]
            else:
                angle_delta = []
            if len(dihedral_label_lst):
                dihedral_delta = dihedral_label_lst[:, -1]
            else:
                dihedral_delta = np.array([])
            
            bond_len = len(bond_delta)
            
            
            

            dihedral_delta = handle_abnormal_torsioin(dihedral_delta)
            delta_r_lst = np.concatenate((bond_delta, angle_delta, dihedral_delta))
            
            # degree to the radian
            delta_r_lst[bond_len:] = np.deg2rad(delta_r_lst[bond_len:])
            
            delta_r_lst = delta_r_lst / (var_lst ** 2) # divide the varaince
            
            noise_target_lst = []
            reg_value_lst = []
            for n_idx in range(self.sample_num):
                noise_target = np.random.normal(loc=np.zeros_like(coords), scale=1)
                noisy_coord = (coords + noise_target * self.dev_sigma) # NOTE sigma is 0.04
                noise_target_lst.append(noise_target)
                
                noisy_mol_new = copy.deepcopy(noise_mol)
                noisy_conf = noisy_mol_new.GetConformer()
                for i in range(atom_num):
                    x,y,z = noisy_coord[i]
                    noisy_conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
                
                
                bond_len_lst_noise, angle_lst_noise, dihedral_lst_noise = get_info_by_gem_idx(noisy_mol_new, edge_idx, angle_idx, dihedral_idx)
                
                dih_len = len(dihedral_lst_noise)
                noise_rlst = np.concatenate((bond_len_lst_noise, angle_lst_noise, dihedral_lst_noise))
                
                noise_diff_lst = noise_rlst - org_rlst
                
                noise_diff_lst = noise_diff_lst / self.derivative_var  # NOTE: divide sigma
                
                # NOTE: fix, only handle torsion part
                if dih_len:
                    noise_diff_lst[-dih_len:] = handle_abnormal_torsioin(noise_diff_lst[-dih_len:])
                
                # degree to the radian
                noise_diff_lst[bond_len:] = np.deg2rad(noise_diff_lst[bond_len:])
                
                
                reg_value_lst.append(float(np.dot(delta_r_lst, noise_diff_lst)))
            

            
            
            org_data.reg_value_lst = reg_value_lst
            org_data.noise_target_lst = np.array(noise_target_lst, dtype=np.float32)
            
            nsample = org_data.noise_target_lst.shape[0]
            aa = org_data.noise_target_lst.reshape((nsample,-1))
            bb = np.asarray(org_data.reg_value_lst)
            ff_learn, residual, rank, singular = np.linalg.lstsq(aa, bb, rcond=None)
            org_data.ff_learn = torch.tensor(ff_learn.reshape((-1, 3)), dtype=torch.float)
            
        elif self.use_force_param:
            # divide the var from param
            bat_var_lst = np.array(bat_var_lst)
            bond_len = len(bond_label_lst)
            angle_len = len(angle_label_lst)
            torsion_len = len(dihedral_label_lst)
            if bond_len > 0:
                bond_label_lst[:,2] = bond_label_lst[:,2] / (bat_var_lst[:bond_len] ** 2)
            if angle_len > 0:
                angle_label_lst[:,3] = angle_label_lst[:,3] / (bat_var_lst[bond_len:bond_len+angle_len] ** 2)
            if torsion_len > 0:
                dihedral_label_lst[:,4] = dihedral_label_lst[:,4] / (bat_var_lst[bond_len+angle_len:] ** 2)
            
            
        # if non_rotate_dihedral_len:
        #     dihedral_label_lst = np.array(dihedral_label_lst, dtype=np.float32)
        #     # dihedral_label_lst[:, 4] = (noise_dihedarl - org_dihedral)[:non_rotate_dihedral_len]

        #     mask = ~np.isnan(dihedral_label_lst[:, 4])
        #     dihedral_label_lst = dihedral_label_lst[mask]
        # if rotate_dihedral_len:
        #     rotate_dihedral_label_lst = np.array(rotate_dihedral_label_lst, dtype=np.float32)
        #     # rotate_dihedral_label_lst[:, 4] = (noise_dihedarl - org_dihedral)[non_rotate_dihedral_len:]

        #     mask = ~np.isnan(rotate_dihedral_label_lst[:, 4])
        #     rotate_dihedral_label_lst = rotate_dihedral_label_lst[mask]

        # set coordinate to atom
        org_data.pos = torch.tensor(coords)
        org_data.bond_target = torch.tensor(bond_label_lst)
        org_data.angle_target = torch.tensor(angle_label_lst)
        org_data.dihedral_target = torch.tensor(dihedral_label_lst)
        org_data.rotate_dihedral_target = torch.tensor(rotate_dihedral_label_lst)
        
        if self.frad_guassian_scale:
            
            noise = torch.randn_like(torch.tensor(coords)) * self.frad_guassian_scale
            data_noise = coords + noise.numpy()
            org_data.pos_frad = torch.tensor(data_noise)    
            org_data.pos_target = noise
        else:
            org_data.pos_target = torch.tensor(coords - org_coords)

        if org_data.pos_target.isnan().sum().item():
            print('data nan')

        return org_data

# equilibrium
EQ_MOL_LST = None
EQ_EN_LST = None

class PCQM4MV2_Dihedral2(PCQM4MV2_XYZ):
    def __init__(self, root: str, sdf_path: str, dihedral_angle_noise_scale: float, position_noise_scale: float, composition: bool, decay=False, decay_coe=0.2, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None, equilibrium=False, eq_weight=False, cod_denoise=False, integrate_coord=False, addh=False, mask_atom=False, mask_ratio=0.15, bat_noise=False):
        assert dataset_arg is None, "PCQM4MV2_Dihedral does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.suppl = Chem.SDMolSupplier(sdf_path)
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.position_noise_scale = position_noise_scale
        self.composition = composition # angle noise as the start

        self.decay = decay
        self.decay_coe = decay_coe

        self.random_pos_prb = 0.5
        self.equilibrium = equilibrium # equilibrium settings
        self.eq_weight = eq_weight
        self.cod_denoise = cod_denoise # reverse to coordinate denoise

        self.integrate_coord = integrate_coord
        self.addh = addh

        self.mask_atom = mask_atom
        self.mask_ratio = mask_ratio
        self.num_atom_type = 119

        self.bat_noise = bat_noise
        
        global MOL_LST
        global EQ_MOL_LST
        global EQ_EN_LST
        if self.equilibrium and EQ_MOL_LST is None:
            # debug
            EQ_MOL_LST = np.load('MG_MOL_All.npy', allow_pickle=True) # mol lst
            EQ_EN_LST = np.load('MG_All.npy', allow_pickle=True) # energy lst
        else:
            if MOL_LST is None:
            # import pickle
            # with open(sdf_path, 'rb') as handle:
            #     MOL_LST = pickle.load(handle)
            # MOL_LST = np.load("mol_iter_all.npy", allow_pickle=True)
                # MOL_LST = np.load("h_mol_lst.npy", allow_pickle=True)
                MOL_LST = lmdb.open('/data/protein/SKData/DenoisingData/MOL_LMDB', readonly=True, subdir=True, lock=False)
            
        if debug:
            global MOL_DEBUG_LST
            if MOL_DEBUG_LST is None:
                # MOL_DEBUG_LST = Chem.SDMolSupplier("pcqm4m-v2-train.sdf")
                MOL_DEBUG_LST = np.load("mol_iter_all.npy", allow_pickle=True)
    
    def transform_noise(self, data, position_noise_scale):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
        data_noise = data + noise.numpy()
        return data_noise
    
    def transform_noise_decay(self, data, position_noise_scale, decay_coe_lst):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale * torch.tensor(decay_coe_lst)
        data_noise = data + noise.numpy()
        return data_noise

    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', Data]:
        org_data = super().__getitem__(idx)
        org_atom_num = org_data.pos.shape[0]
        # change org_data coordinate
        # get mol

        # check whether mask or not
        if self.mask_atom:
            num_atoms = org_data.z.size(0)
            sample_size = int(num_atoms * self.mask_ratio + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)
            org_data.mask_node_label = org_data.z[masked_atom_indices]
            org_data.z[masked_atom_indices] = self.num_atom_type
            org_data.masked_atom_indices = torch.tensor(masked_atom_indices)

        if self.equilibrium:
            # for debug
            # max_len = 422325 - 1
            # idx = idx.item() % max_len
            idx = idx.item()
            mol = copy.copy(EQ_MOL_LST[idx])
            energy_lst = EQ_EN_LST[idx]
            eq_confs = len(energy_lst)
            conf_num = mol.GetNumConformers()
            assert conf_num == (eq_confs + 1)
            if eq_confs:
                weights = F.softmax(-torch.tensor(energy_lst))
                # random pick one
                pick_lst = [idx for idx in range(conf_num)]
                p_idx = random.choice(pick_lst)
                
                for conf_id in range(conf_num):
                    if conf_id != p_idx:
                        mol.RemoveConformer(conf_id)
                # only left p_idx
                if p_idx == 0:
                    weight = 1
                else:
                    if self.eq_weight:
                        weight = 1
                    else:
                        weight = weights[p_idx - 1].item()
                        
            else:
                weight = 1
            
        else:
            ky = str(idx.item()).encode()
            serialized_data = MOL_LST.begin().get(ky)
            mol = pickle.loads(serialized_data)
            # mol = MOL_LST[idx.item()]


        atom_num = mol.GetNumAtoms()

        # get rotate bond
        if self.addh:
            rotable_bonds = get_torsions([mol])
        else:
            no_h_mol = Chem.RemoveHs(mol)
            rotable_bonds = get_torsions([no_h_mol])
        

        # prob = random.random()
        cod_denoise = self.cod_denoise
        if self.integrate_coord:
            assert not self.cod_denoise
            prob = random.random()
            if prob < 0.5:
                cod_denoise = True
            else:
                cod_denoise = False

        if atom_num != org_atom_num or len(rotable_bonds) == 0 or cod_denoise: # or prob < self.random_pos_prb:
            pos_noise_coords = self.transform_noise(org_data.pos, self.position_noise_scale)
            org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
            org_data.pos = torch.tensor(pos_noise_coords)

            
            if self.equilibrium:
                org_data.w1 = weight
                org_data.wg = torch.tensor([weight for _ in range(org_atom_num)], dtype=torch.float32)
            return org_data

        # else angel random
        # if len(rotable_bonds):
        org_angle = []
        if self.decay:
            rotate_bonds_order, rb_depth = get_rotate_order_info(mol, rotable_bonds)
            decay_coe_lst = []
            for i, rot_bond in enumerate(rotate_bonds_order):
                org_angle.append(GetDihedral(mol.GetConformer(), rot_bond))
                decay_scale = (self.decay_coe) ** (rb_depth[i] - 1)    
                decay_coe_lst.append(self.dihedral_angle_noise_scale*decay_scale)
            noise_angle = self.transform_noise_decay(org_angle, self.dihedral_angle_noise_scale, decay_coe_lst)
            new_mol = apply_changes(mol, noise_angle, rotate_bonds_order)
        else:
            if self.bat_noise:
                new_mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst = add_equi_noise_new(mol, add_ring_noise=False)
            else:
                for rot_bond in rotable_bonds:
                    org_angle.append(GetDihedral(mol.GetConformer(), rot_bond))
                org_angle = np.array(org_angle)        
                noise_angle = self.transform_noise(org_angle, self.dihedral_angle_noise_scale)
                new_mol = apply_changes(mol, noise_angle, rotable_bonds)
        
        coord_conf = new_mol.GetConformer()
        pos_noise_coords_angle = np.zeros((atom_num, 3), dtype=np.float32)
        # pos_noise_coords = new_mol.GetConformer().GetPositions()
        for idx in range(atom_num):
            c_pos = coord_conf.GetAtomPosition(idx)
            pos_noise_coords_angle[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]

        # coords = np.zeros((atom_num, 3), dtype=np.float32)
        # coord_conf = mol.GetConformer()
        # for idx in range(atom_num):
        #     c_pos = coord_conf.GetAtomPosition(idx)
        #     coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
        # coords = mol.GetConformer().GetPositions()

        if self.bat_noise:
            # check nan
            if torch.tensor(pos_noise_coords_angle).isnan().sum().item():# contains nan
                print('--------bat nan, revert back to org coord-----------')
                pos_noise_coords_angle = org_data.pos.numpy()



        pos_noise_coords = self.transform_noise(pos_noise_coords_angle, self.position_noise_scale)
        
        
        # if self.composition or not len(rotable_bonds):
        #     pos_noise_coords = self.transform_noise(coords, self.position_noise_scale)
        #     if len(rotable_bonds): # set coords into the mol
        #         conf = mol.GetConformer()
        #         for i in range(mol.GetNumAtoms()):
        #             x,y,z = pos_noise_coords[i]
        #             conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        

        

        
        # org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
        if self.composition:
            org_data.pos_target = torch.tensor(pos_noise_coords - pos_noise_coords_angle)
            org_data.pos = torch.tensor(pos_noise_coords)
        else:
            org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
            org_data.pos = torch.tensor(pos_noise_coords)
        
        if self.equilibrium:
            org_data.w1 = weight
            org_data.wg = torch.tensor([weight for _ in range(atom_num)], dtype=torch.float32)

        return org_data






# S_IDX = None
# coordinate denoise and (bond, angle, dehedral angle) denoise
class PCQM4MV2_DihedralAll(PCQM4MV2_XYZ):
    def __init__(self, root: str, sdf_path: str, dihedral_angle_noise_scale: float, position_noise_scale: float, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None, addh=False):
        assert dataset_arg is None, "PCQM4MV2_DihedralAll does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.suppl = Chem.SDMolSupplier(sdf_path)
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.position_noise_scale = position_noise_scale
        # self.composition = composition # angle noise as the start


        # self.integrate_coord = integrate_coord
        self.addh = addh

        self.num_atom_type = 119
        global MOL_LST
        # , S_IDX
        if MOL_LST is None:
            MOL_LST = np.load("h_mol_lst.npy", allow_pickle=True)
        # if S_IDX is None:
        #     S_IDX = np.load("PCQM_valid.npy")
            
    
    def transform_noise(self, data, position_noise_scale):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
        data_noise = data + noise.numpy()
        return data_noise
    
    def filter_nan(self, idx_array, noise_array):
        valid_idx = ~np.isnan(noise_array)
        return idx_array[valid_idx], noise_array[valid_idx]

    def _concat_idx_label(self, edge_idx, angle_idx, dihedral_idx, noise_bond_len_label, noise_angle_label, noise_dihedral_label):
        edge_idx = np.array(edge_idx)
        angle_idx = np.array(angle_idx)
        dihedral_idx = np.array(dihedral_idx)

        edge_idx , noise_bond_len_label = self.filter_nan(edge_idx, noise_bond_len_label)
        angle_idx , noise_angle_label = self.filter_nan(angle_idx, noise_angle_label)
        dihedral_idx, noise_dihedral_label = self.filter_nan(dihedral_idx, noise_dihedral_label)

        
        # handle dihedral noise
        # noise_dihedral_label[noise_dihedral_label > 180] -= 360
        # noise_dihedral_label[noise_dihedral_label < -180] += 360

        # try:
        if edge_idx.size == 0:
            edge_res = np.array([])
        else:
            edge_res = np.concatenate([edge_idx, noise_bond_len_label.reshape([-1, 1])], axis=1)
        if angle_idx.size == 0:
            angle_res = np.array([])
        else:
            angle_res = np.concatenate([angle_idx, noise_angle_label.reshape([-1, 1])], axis=1)
        if dihedral_idx.size == 0:
            dihedral_res = np.array([])
        else:
            dihedral_res = np.concatenate([dihedral_idx, noise_dihedral_label.reshape([-1, 1])], axis=1)
        # except Exception as e:
        #     print(e)
        return edge_res, angle_res, dihedral_res
        # return np.concatenate([edge_idx, noise_bond_len_label.reshape([-1, 1])], axis=1),\
        #        np.concatenate([angle_idx, noise_angle_label.reshape([-1, 1])], axis=1),\
        #        np.concatenate([dihedral_idx, noise_dihedral_label.reshape([-1, 1])], axis=1)


    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', Data]:
        org_data = super().__getitem__(idx)
        org_atom_num = org_data.pos.shape[0]
        # change org_data coordinate
        # get mol coordinate

        mol = MOL_LST[idx.item()]


        atom_num = mol.GetNumAtoms()
        

        # if atom_num != org_atom_num:
        #     idx = np.random.choice(S_IDX)
        #     mol = MOL_LST[idx]
        #     atom_num = mol.GetNumAtoms()
        
        if atom_num != org_atom_num:
            print('assert atom_num == org_atom_num failed')
            atoms = mol.GetAtoms()
            z_lst = []
            for i in range(atom_num):
                atom = atoms[i]
                z_lst.append(atom.GetAtomicNum()) # atomic num start from 1
            
            org_data.z = torch.tensor(z_lst) # atomic num start from 1

            coord_conf = mol.GetConformer()
            new_pos = np.zeros((atom_num, 3), dtype=np.float32)
            # pos_noise_coords = new_mol.GetConformer().GetPositions()
            for idx in range(atom_num):
                c_pos = coord_conf.GetAtomPosition(idx)
                new_pos[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
            org_data.pos = torch.tensor(new_pos)


        rotable_bonds = get_torsions([mol])
        # assert atom_num == org_atom_num

        edge_idx, angle_idx, dihedral_idx = get_2d_gem(mol)

        if len(rotable_bonds) == 0: # or cod_denoise: # or prob < self.random_pos_prb:
            # NOTE: get original mol bond angle dihedral info
            try:
                bond_len_lst, angle_lst, dihedral_lst = get_info_by_gem_idx(mol, edge_idx, angle_idx, dihedral_idx)
            except Exception as e:
                print(f'error : {e}')
            pos_noise_coords = self.transform_noise(org_data.pos, self.position_noise_scale)

            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                x,y,z = pos_noise_coords[i]
                conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
            
            noise_bond_len_lst, noise_angle_lst, noise_dihedral_lst = get_info_by_gem_idx(mol, edge_idx, angle_idx, dihedral_idx)
            
            bond_label_lst, angle_label_lst, dihedral_label_lst = self._concat_idx_label(edge_idx, angle_idx, dihedral_idx, noise_bond_len_lst-bond_len_lst, noise_angle_lst-angle_lst, noise_dihedral_lst-dihedral_lst)

            org_data.bond_target = torch.tensor(bond_label_lst)
            org_data.angle_target = torch.tensor(angle_label_lst)
            org_data.dihedral_target = torch.tensor(dihedral_label_lst)

            org_data.pos_target = torch.tensor(pos_noise_coords - org_data.pos.numpy())
            org_data.pos = torch.tensor(pos_noise_coords)
            # if self.equilibrium:
            #     org_data.w1 = weight
            #     org_data.wg = torch.tensor([weight for _ in range(org_atom_num)], dtype=torch.float32)
            return org_data

        # else angel random
        # if len(rotable_bonds):
        org_angle = []
        for rot_bond in rotable_bonds:
            org_angle.append(GetDihedral(mol.GetConformer(), rot_bond))
        org_angle = np.array(org_angle)        
        noise_angle = self.transform_noise(org_angle, self.dihedral_angle_noise_scale)
        new_mol = apply_changes(mol, noise_angle, rotable_bonds)
        
        
        bond_len_lst, angle_lst, dihedral_lst = get_info_by_gem_idx(new_mol, edge_idx, angle_idx, dihedral_idx)


        coord_conf = new_mol.GetConformer()
        pos_noise_coords_angle = np.zeros((atom_num, 3), dtype=np.float32)
        # pos_noise_coords = new_mol.GetConformer().GetPositions()
        for idx in range(atom_num):
            c_pos = coord_conf.GetAtomPosition(idx)
            pos_noise_coords_angle[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]


        pos_noise_coords = self.transform_noise(pos_noise_coords_angle, self.position_noise_scale)
        
        conf = new_mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            x,y,z = pos_noise_coords[i]
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        noise_bond_len_lst, noise_angle_lst, noise_dihedral_lst = get_info_by_gem_idx(new_mol, edge_idx, angle_idx, dihedral_idx)

        
        bond_label_lst, angle_label_lst, dihedral_label_lst = self._concat_idx_label(edge_idx, angle_idx, dihedral_idx, noise_bond_len_lst-bond_len_lst, noise_angle_lst-angle_lst, noise_dihedral_lst-dihedral_lst)

        org_data.bond_target = torch.tensor(bond_label_lst, dtype=torch.float32)
        org_data.angle_target = torch.tensor(angle_label_lst, dtype=torch.float32)
        org_data.dihedral_target = torch.tensor(dihedral_label_lst, dtype=torch.float32)
        # org_data.rotate_dihedral_target = torch.tensor(rotate_dihedral_label_lst)
        

        org_data.pos_target = torch.tensor(pos_noise_coords - pos_noise_coords_angle)
        org_data.pos = torch.tensor(pos_noise_coords)




        return org_data




# learn force field exp
ORG_MOLS = None
SAMPLE_POS = None
FORCES_LABEL = None

# exp1: noisy node --> dft force feild
# exp2: noisy node --> noisy
# exp3: frad noisy node


# exp4: use rkdit conformation: frad or coord denoise
class PCQM4MV2_DihedralF(PCQM4MV2_XYZ):
    def __init__(self, root: str, sdf_path: str, dihedral_angle_noise_scale: float, position_noise_scale: float, composition: bool, force_field: bool=False, pred_noise: bool=False, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None, cod_denoise=False, rdkit_conf=False):
        assert dataset_arg is None, "PCQM4MV2_Dihedral does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.suppl = Chem.SDMolSupplier(sdf_path)
        self.dihedral_angle_noise_scale = dihedral_angle_noise_scale
        self.position_noise_scale = position_noise_scale
        self.composition = composition # compose dihedral angle and position noise
        self.force_field = force_field
        self.pred_noise = pred_noise

        self.rdkit_conf = rdkit_conf

        
        global ORG_MOLS
        global SAMPLE_POS
        global FORCES_LABEL
        if ORG_MOLS is None:
            if self.rdkit_conf:
                ORG_MOLS = np.load('/home/fengshikun/Pretraining-Denoising/rdkit_mols_conf_lst.npy', allow_pickle=True)    
            else:
                ORG_MOLS = np.load('/home/fengshikun/Backup/Denoising/data/dft/head_1w/mols_head_1w.npy', allow_pickle=True)
                SAMPLE_POS = np.load('/home/fengshikun/Backup/Denoising/data/dft/head_1w/mols_head_1w_pos.npy', allow_pickle=True)
                FORCES_LABEL = np.load('/home/fengshikun/Backup/Denoising/data/dft/head_1w/mols_head_1w_force.npy', allow_pickle=True)
        self.mol_num = len(ORG_MOLS)
        self.cod_denoise = cod_denoise
        print(f'load PCQM4MV2_DihedralF complete, mol num is {self.mol_num}')
    
    def process_data(self, max_node_num=30):
        pass


    def __len__(self):
        return self.mol_num

    
    def transform_noise(self, data, position_noise_scale):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
        data_noise = data + noise.numpy()
        return data_noise

    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union['Dataset', Data]:
        org_data = super().__getitem__(idx)
        org_mol = ORG_MOLS[idx.item()]
        # get org pos

        org_atom_num = org_data.pos.shape[0]

        atom_num = org_mol.GetNumAtoms()

        
        # assert org_atom_num == atom_num
        org_pos = np.zeros((atom_num, 3), dtype=np.float32)
        # pos_noise_coords = new_mol.GetConformer().GetPositions()
        
        # check the conformers number(when use the rdkit generated conformers, conf number may be zero)
        conf_num = org_mol.GetNumConformers()
        if not conf_num:
            # use the orginal pos
            assert self.rdkit_conf # this only happen when use rdkit generated conf
            org_pos = org_data.pos
            coord_conf = Chem.Conformer(org_mol.GetNumAtoms())

            if org_atom_num != atom_num:
                pos_noise_coords = self.transform_noise(org_pos, self.position_noise_scale)
                org_data.pos_target = torch.tensor(pos_noise_coords - org_pos)
                org_data.pos = torch.tensor(pos_noise_coords)
                return org_data

            for i in range(atom_num):
                coord_conf.SetAtomPosition(i, (org_pos[i][0].item(), org_pos[i][1].item(), org_pos[i][2].item()))
            org_mol.AddConformer(coord_conf)
        else:
            coord_conf = org_mol.GetConformer()
            atoms = org_mol.GetAtoms()
            z_lst = [] # the force filed data may not consistant with the original data with same index. we only pick mols which have less atoms than 30 atoms.
            for i in range(atom_num):
                c_pos = coord_conf.GetAtomPosition(i)
                org_pos[i] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
                atom = atoms[i]
                z_lst.append(atom.GetAtomicNum()) # atomic num start from 1
            
            org_data.z = torch.tensor(z_lst) # atomic num start from 1
        # random sample one pos
        if self.force_field or self.pred_noise:
            sample_poses = SAMPLE_POS[idx]
            sample_pos_num = len(sample_poses)
            random_idx = random.randint(0, sample_pos_num - 1)
            sample_pos = sample_poses[random_idx]
            
            force_label = FORCES_LABEL[idx][random_idx]

        if self.force_field:
            org_data.pos_target = torch.tensor(force_label)
            org_data.pos = torch.tensor(sample_pos)
        elif self.pred_noise:
            org_data.pos_target = torch.tensor(sample_pos - org_pos)
            org_data.pos = torch.tensor(sample_pos)
        elif self.composition:
            rotable_bonds = get_torsions([org_mol])
            if len(rotable_bonds) == 0 or self.cod_denoise:
                pos_noise_coords = self.transform_noise(org_pos, self.position_noise_scale)
                org_data.pos_target = torch.tensor(pos_noise_coords - org_pos)
                org_data.pos = torch.tensor(pos_noise_coords)
                return org_data

            org_angle = []
            for rot_bond in rotable_bonds:
                org_angle.append(GetDihedral(org_mol.GetConformer(), rot_bond))
            org_angle = np.array(org_angle)        
            noise_angle = self.transform_noise(org_angle, self.dihedral_angle_noise_scale)
            new_mol = apply_changes(org_mol, noise_angle, rotable_bonds)
        
            coord_conf = new_mol.GetConformer()
            pos_noise_coords_angle = np.zeros((atom_num, 3), dtype=np.float32)
            # pos_noise_coords = new_mol.GetConformer().GetPositions()
            for idx in range(atom_num):
                c_pos = coord_conf.GetAtomPosition(idx)
                pos_noise_coords_angle[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]

            pos_noise_coords = self.transform_noise(pos_noise_coords_angle, self.position_noise_scale)
            org_data.pos_target = torch.tensor(pos_noise_coords - pos_noise_coords_angle)
            org_data.pos = torch.tensor(pos_noise_coords)
        else:
            raise Exception('Not implemented situation, one of pred_noise, composition and force_filed should be true')
        
        return org_data



class PCQM4MV2_3D:
    """Data loader for PCQM4MV2 from raw xyz files.
    
    Loads data given a path with .xyz files.
    """
    
    def __init__(self, path) -> None:
        self.path = path
        self.xyz_files = glob.glob(path + '/*/*.xyz')
        self.xyz_files = sorted(self.xyz_files, key=self._molecule_id_from_file)
        self.num_molecules = len(self.xyz_files)
        
    def read_xyz_file(self, file_path):
        atom_types = np.genfromtxt(file_path, skip_header=1, usecols=range(1), dtype=str)
        atom_types = np.array([ase.Atom(sym).number for sym in atom_types])
        atom_positions = np.genfromtxt(file_path, skip_header=1, usecols=range(1, 4), dtype=np.float32)        
        return {'atom_type': atom_types, 'coords': atom_positions}
    
    def _molecule_id_from_file(self, file_path):
        return int(os.path.splitext(os.path.basename(file_path))[0])
    
    def __len__(self):
        return self.num_molecules
    
    def __getitem__(self, idx):
        return self.read_xyz_file(self.xyz_files[idx])