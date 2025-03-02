# %%
import os
import pickle
import lmdb
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  
import warnings
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool
from rdkit.Chem import rdDistGeom

# %%
def print_failure_causes(counts):
    for i,k in enumerate(rdDistGeom.EmbedFailureCauses.names):
        print(k,counts[i])
    # in v2022.03.1 two names are missing from `rdDistGeom.EmbedFailureCauses`:
    print('LINEAR_DOUBLE_BOND',counts[i+1])
    print('BAD_DOUBLE_BOND_STEREO',counts[i+2])    

# %%

def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates


def smi2_3Dcoords(smi,cnt):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list=[]
    seed  = 0
    success_num = 0
    try_num = 0
    MAX_TRY = 50
    ps = AllChem.ETKDGv3()
    ps.maxIterations = 5000
    
    ps_fast = AllChem.ETKDGv3()
    
    
    while success_num < cnt and try_num < MAX_TRY:
        seed = try_num
        try_num += 1

        ps.randomSeed = seed        
        ps_fast.randomSeed = seed

        try:
            res = AllChem.EmbedMolecule(mol, ps_fast)  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)       # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    continue
                    # print("Failed to generate 3D, replace with 2D")
                    # coordinates = smi2_2Dcoords(smi)            
                    
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, ps)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)       # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    continue
                    # print("Failed to generate 3D, replace with 2D")
                    # coordinates = smi2_2Dcoords(smi) 
        except:
            continue
            print("Failed to generate 3D, replace with 2D")
            # coordinates = smi2_2Dcoords(smi) 

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
        success_num += 1
    if success_num < cnt:
        print("failed to generate 3D coordinates for",smi, "use 2D coordinates instead")
        for _ in range(cnt - success_num):
            coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    print("success_num:",success_num, "; try_num:",try_num)
    return coordinate_list


def inner_smi2coords(content):
    smi = content[0]
    target = content[1:]
    cnt = 10 # conformer num,all==11, 10 3d + 1 2d

    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > 400:
        coordinate_list =  [smi2_2Dcoords(smi)] * (cnt+1)
        print("atom num >400,use 2D coords",smi)
    else:
        coordinate_list = smi2_3Dcoords(smi,cnt)
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H 
    return pickle.dumps({'atoms': atoms, 
    'coordinates': coordinate_list, 
    'mol':mol,'smi': smi, 'target': target}, protocol=-1)


def smi2coords(content):
    try:
        return inner_smi2coords(content)
    except:
        print("failed smiles: {}".format(content[0]))
        return None


def write_lmdb(inpath='./', outpath='./', nthreads=16):

    df = pd.read_csv(os.path.join(inpath))
    mol_col = "mol"
    target_col = "RLU (log2)"
    df = df[[mol_col, target_col]]
    sz = len(df)
    train, valid, test = df[:int(sz*0.8)], df[int(sz*0.8):int(sz*0.9)], df[int(sz*0.9):]
    # train, valid, test = df[:int(sz*0.9)], df[int(sz*0.9):int(sz)], df[int(sz):]
    for name, content_list in [
        ('train.lmdb', zip(*[train[c].values.tolist() for c in train])),
                                ('valid.lmdb', zip(*[valid[c].values.tolist() for c in valid])),
                                ('test.lmdb', zip(*[test[c].values.tolist() for c in test]))]:
        os.makedirs(outpath, exist_ok=True)
        output_name = os.path.join(outpath, name)
        try:
            os.remove(output_name)
        except:
            pass
        env_new = lmdb.open(
            output_name,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100e9),
        )
        txn_write = env_new.begin(write=True)
        with Pool(nthreads) as pool:
            i = 0
            for inner_output in tqdm(pool.imap(smi2coords, content_list)):
                if inner_output is not None:
                    txn_write.put(f'{i}'.encode("ascii"), inner_output)
                    i += 1
            print('{} process {} lines'.format(name, i))
            txn_write.commit()
            env_new.close()

write_lmdb(inpath='/h/pangkuan/dev/SDL-LNP/model/4CR-1920.csv', outpath='/scratch/ssd004/datasets/cellxgene/3d_molecule_data/1920-lib', nthreads=64)


