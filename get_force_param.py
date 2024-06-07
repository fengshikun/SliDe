from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.topology import Molecule, Topology
import numpy as np
import lmdb
import pickle
from tqdm import tqdm

forcefield = ForceField("openff-2.0.0.offxml")

MOL_LST = lmdb.open('/data/protein/SKData/DenoisingData/MOL_LMDB', readonly=True, subdir=True, lock=False)

para_env = lmdb.open('/data/protein/SKData/DenoisingData/Param_DB', map_size=1099511627776) 

with MOL_LST.begin() as txn:
    _keys = list(txn.cursor().iternext(values=False))

txnw = para_env.begin(write=True)
for ky in tqdm(_keys):
    serialized_data = MOL_LST.begin().get(ky)
    serialized_data = MOL_LST.begin().get(ky)
    mol = pickle.loads(serialized_data)
    try:
        equi_mol = Molecule.from_rdkit(mol)    
        topology = Topology.from_molecules(molecules=[equi_mol])
        para = forcefield.label_molecules(topology)
        # print('save_para')
        serialized_data_para = pickle.dumps(para)
        txnw.put(ky, serialized_data_para)
    except Exception as e:
        print(f'exeption captured {e}, at key {ky}')
txnw.commit()
para_env.close()
#从para中提取BAT 的parameter:
# bond_para = para[0]["Bonds"][(i_idx, j_idx)].k._value # for bond i-j
# angle_para = para[0]["Angles"][(i_idx, j_idx, k_idx)].k._value #for angle i-j-k
# torsion_para = para[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].k[0]._value 
# torsion_period=para[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].periodicity[0]  #for torsion i-j-k-l

# #把parameter转化为noise的std:
# std_bond=np.sqrt(1/bond_para)
# std_angle=np.sqrt(1/angle_para) #angle values should in radian
# std_torsion=np.sqrt(1/(torsion_para*torsion_period^2))

#  附：toolkit documentation: https://docs.openforcefield.org/projects/toolkit/en/stable/api/generated/openff.toolkit.typing.engines.smirnoff.ForceField.html
# 物理含义的解释：https://openforcefield.github.io/standards/standards/smirnoff/#physical-constants