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
import argparse

# %%

def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates


def smi2_3Dcoords(smi, cnt, MAX_TRY = 50, max_iteration=5000):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list=[]
    seed  = 0
    success_num = 0
    try_num = 0
    ps = AllChem.ETKDGv3()
    ps.maxIterations = max_iteration
    
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
    'smi': smi, 'target': target}, protocol=-1)


def smi2coords(content):
    try:
        return inner_smi2coords(content)
    except:
        print("failed smiles: {}".format(content[0]))
        return None

def smi2coords_smi(smi):
    try:
        cnt = 10 # conformer num,all==11, 10 3d + 1 2d

        mol = Chem.MolFromSmiles(smi)
        if len(mol.GetAtoms()) > 400:
            coordinate_list =  [smi2_2Dcoords(smi)] * (cnt+1)
            print("atom num >400,use 2D coords",smi)
        else:
            coordinate_list = smi2_3Dcoords(smi,cnt, MAX_TRY = 15, max_iteration=4000)
            coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
            assert len(coordinate_list) == 11; "coordinate_list is not 11 length"
        mol = AllChem.AddHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H 
        return smi, pickle.dumps({'atoms': atoms, 
                        'coordinates': coordinate_list, 
                        'smi': smi}, protocol=-1)
    except:
        print("failed smiles: {}".format(smi))
        return None


def smi2coords_pretrain(smi):
    try:
        cnt = 10 # conformer num,all==11, 10 3d + 1 2d

        mol = Chem.MolFromSmiles(smi)
        if len(mol.GetAtoms()) > 400:
            coordinate_list =  [smi2_2Dcoords(smi)] * (cnt+1)
            print("atom num >400,use 2D coords",smi)
        else:
            coordinate_list = smi2_3Dcoords(smi,cnt, MAX_TRY = 15, max_iteration=4000)
            coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
        mol = AllChem.AddHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H 
        return smi, pickle.dumps({
                        'atoms': atoms, 
                        'coordinates': coordinate_list,
                        'smi': smi}, protocol=-1)
    except:
        print("failed smiles: {}".format(smi))
        return None


def write_lmdb(inpath='./', outpath='./', nthreads=16):

    df = pd.read_csv(os.path.join(inpath))
    mol_col = "combined_mol_SMILES"
    df = df[[mol_col, ]]
    os.makedirs(outpath, exist_ok=True)
    name = 'test.lmdb'
    output_name = os.path.join(outpath, name)
    values = df.values.tolist()
    content_list = [(x[-1], -1,) for x in values]
    smi_name_list = []
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
                smi_name_list.append(content_list[i][0])
                i += 1
            else:
                print("Receiving None: {}".format(i))
        txn_write.commit()
        env_new.close()
        # save the smi_name_list txt
    with open(os.path.join(outpath, 'smi_name_list.txt'), 'w') as f:
        for item in smi_name_list:
            f.write("%s\n" % item)



def write_lmdb_missing(inpath='./', outpath='./', nthreads=16):

    """Read text from inpath and write to lmdb in outpath
    """
    with open(inpath, 'r') as f:
        smiles = f.readlines()
    os.makedirs(outpath, exist_ok=True)
    name = 'map.lmdb'
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
    smi_name_list = []
    txn_write = env_new.begin(write=True)
    with Pool(nthreads) as pool:
        i = 0
        for smi, inner_output in tqdm(pool.imap(smi2coords_smi, smiles)):
            if inner_output is not None:
                txn_write.put(str(smi).encode("ascii"), inner_output)
                smi_name_list.append(smi)
                i += 1
            else:
                print("Receiving None: {}".format(smi))
        print('{} process {} lines'.format(name, i))
        txn_write.commit()
        env_new.close()
    # save the smi_name_list txt
    with open(os.path.join(outpath, 'smi_name_list.txt'), 'w') as f:
        for item in smi_name_list:
            f.write("%s\n" % item)
    

def write_lmdb_extended_lipid(inpath='./', outpath='./', nthreads=16):

    """Read text from inpath and write to lmdb in outpath
    """
    with open(inpath, 'r') as f:
        smiles = f.readlines()
    os.makedirs(outpath, exist_ok=True)
    name = 'train.lmdb'
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
        map_size=int(500e9),
    )
    txn_write = env_new.begin(write=True)
    with Pool(nthreads) as pool:
        i = 0
        for smi, inner_output in tqdm(pool.imap(smi2coords_pretrain, smiles)):
            if inner_output is not None:
                txn_write.put(str(smi).encode("ascii"), inner_output)
                i += 1
        print('{} process {} lines'.format(name, i))
        txn_write.commit()
        env_new.close()
    


parser = argparse.ArgumentParser(description='write_lmdb')
parser.add_argument('--inpath', type=str)
parser.add_argument('--outpath', type=str)
parser.add_argument('--data-type', type=str)

args=parser.parse_args()


cpus_reserved = int(os.environ['SLURM_CPUS_ON_NODE'])
print(f"getting {cpus_reserved} cores")
cpu_count = cpus_reserved

if args.data_type == "1920":
    write_lmdb(inpath=args.inpath,
           outpath=args.outpath,
           nthreads=cpu_count,)
elif args.data_type == "220k":
    write_lmdb(inpath=args.inpath,
           outpath=args.outpath,
           nthreads=cpu_count,)

elif args.data_type == "missing":
    write_lmdb_missing(inpath=args.inpath,
                       outpath=args.outpath,
                        nthreads=cpu_count,)
elif args.data_type == "extended_lipid":
    write_lmdb_extended_lipid(
        inpath=args.inpath,
        outpath=args.outpath,
        nthreads=cpu_count,
    )
else:
    print("data_type not supported")