import pandas as pd
import random
import deepchem as dc
from typing import Iterable
from functools import partial
from torch.utils.data import DataLoader
from dgllife.utils import smiles_to_bigraph, mol_to_bigraph
from rdkit.Chem.PandasTools import LoadSDF
from csv_dataset import MoleculeCSVDataset
from featurizer import *
from utils import *

class MolecularDataLoader(Iterable):
    """
    Data loader for molecular data.
    Parameters
    ----------
    data
        Panda's DataFrame.
    task_names
        Names of label columns. Each column corresponds to a task.
    mol_column
        Name of the column that contains the molecule info.
    message_hidden_feats
        Number of hidden features.
    mol_as_smiles
        Whether the molecules are represented as Smiles.
    node_featurizer
        Callable node featurizing function.
    edge_featurizer
        Callable edge featurizing function.
    batch_size
        Size of train/eval/test batch
    shuffle
        Whether to shuffle the data
    train_val_test
        A tuple of 3 specifying the sizes of data splitting
    """    
    def __init__(self, 
                data, 
                task_names,
                mol_column,
                cache_file_path,
                mol_as_smiles,
                node_featurizer=CanonicalAtomFeaturizer,
                edge_featurizer=CanonicalBondFeaturizer,
                batch_size=8,
                shuffle=True):
            
        self.df = data
        self.tasks = task_names     
        self.shuffle = shuffle  
        self.batch_size = batch_size
        if mol_as_smiles:
            to_graph_func = smiles_to_bigraph
        else:
            to_graph_func = mol_to_bigraph
        self.dataset = MoleculeCSVDataset(df=self.df,
                                        to_graph_func=to_graph_func,
                                        cache_file_path=cache_file_path,
                                        node_featurizer=node_featurizer(atom_data_field='x'),
                                        edge_featurizer=edge_featurizer(bond_data_field='edge_attr'),
                                        mol_column=mol_column,
                                        task_names=task_names,
                                        mol_as_smiles=mol_as_smiles)

        _, graph, _, _ = self.dataset[5]
        self.num_node_attrs = graph.ndata['x'].shape[1]
        self.num_edge_attrs = graph.edata['edge_attr'].shape[1]

        self.dataloader = DataLoader(self.dataset,
                                    collate_fn=collate_molgraphs,
                                    batch_size=self.batch_size,
                                    shuffle=self.shuffle)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataloader)

    def get_splits(self, train_val_test=(0.7,0.1,0.2)):

        if len(train_val_test) != 3:
            raise Exception("train_val_test must be None or a tuple of 3 values.")
        if sum(train_val_test) != 1:
            raise Exception("Sum of train, val, and test proportions must be 1.")
        idx = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(idx)
        train_idx = idx[0:int(train_val_test[0]*len(idx))]
        val_idx = idx[int(train_val_test[0]*len(idx)):int(train_val_test[0]*len(idx))+int(train_val_test[1]*len(idx))]
        test_idx = idx[int(train_val_test[0]*len(idx))+int(train_val_test[1]*len(idx)):]
        train_loader = DataLoader([self.dataset[i] for i in train_idx],
                                        collate_fn=collate_molgraphs,
                                        batch_size=self.batch_size,
                                        shuffle=self.shuffle)
        val_loader = DataLoader([self.dataset[i] for i in val_idx],
                                        collate_fn=collate_molgraphs,
                                        batch_size=self.batch_size,
                                        shuffle=self.shuffle)
        test_loader = DataLoader([self.dataset[i] for i in test_idx],
                                        collate_fn=collate_molgraphs,
                                        batch_size=self.batch_size,
                                        shuffle=self.shuffle)
        return train_loader, val_loader, test_loader

    def get_folds(self, nfolds=5):
        pass

    def get_num_tasks(self):
        return len(self.tasks)

    def get_num_node_attrs(self):
        return self.num_node_attrs

    def get_num_edge_attrs(self):
        return self.num_edge_attrs

import os
CACHE_FOLDER = os.path.abspath(os.getcwd())

from tdc.single_pred import ADME

# Caco2 dataset

def get_caco2():
    data_df = ADME(name = 'Caco2_Wang').get_data()
    return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

def get_Caco2_dataloader(node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    return MolecularDataLoader(data=get_caco2(),
                                task_names=['Y'],
                                mol_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/Caco2.bin',
                                mol_as_smiles=True,
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

# HIA dataset

def get_hia():
    data_df = ADME(name = 'HIA_Hou').get_data()
    return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

def get_HIA_dataloader(node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    return MolecularDataLoader(data=get_hia(),
                                task_names=['Y'],
                                mol_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/HIA.bin',
                                mol_as_smiles=True,
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

# Pgb dataset

def get_pgb():
    data_df = ADME(name = 'Pgp_Broccatelli').get_data()
    return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

def get_HIA_dataloader(node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    return MolecularDataLoader(data=get_pgb(),
                                task_names=['Y'],
                                mol_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/Pgb.bin',
                                mol_as_smiles=True,
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

# LIPO dataset

def get_lipo():
    data_df = ADME(name = 'Lipophilicity_AstraZeneca').get_data()
    return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

def get_LIPO_dataloader(node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    return MolecularDataLoader(data=get_lipo(),
                                task_names=['Y'],
                                mol_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/LIPO.bin',
                                mol_as_smiles=True,
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

# AqSol dataset

def get_aqsol():
    data_df = ADME(name = 'Solubility_AqSolDB').get_data()
    return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

def get_AqSol_dataloader(node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    return MolecularDataLoader(data=get_aqsol(),
                                task_names=['Y'],
                                mol_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/AqSol.bin',
                                mol_as_smiles=True,
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

# FreeSolve

def get_freesolv():
    data_df = ADME(name = 'HydrationFreeEnergy_FreeSolv').get_data()
    return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

def get_FreeSolv_dataloader(node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    return MolecularDataLoader(data=get_freesolv(),
                                task_names=['Y'],
                                mol_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/FreeSolv.bin',
                                mol_as_smiles=True,
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

# BBBP dataset

#TODO: MolNet or TDCommons?

# PPBR dataset

def get_ppbr():
    data_df = ADME(name = 'PPBR_AZ').get_data()
    return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

def get_PPBR_dataloader(node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    return MolecularDataLoader(data=get_ppbr(),
                                task_names=['Y'],
                                mol_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/PPBR.bin',
                                mol_as_smiles=True,
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

# VDss dataset

def get_vdss():
    data_df = ADME(name = 'VDss_Lombardo').get_data()
    return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

def get_VDss_dataloader(node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    return MolecularDataLoader(data=get_vdss(),
                                task_names=['Y'],
                                mol_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/VDss.bin',
                                mol_as_smiles=True,
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Tox

# Tox21

tox21_tasks = retrieve_label_name_list('Tox21')
def get_tox21(tasks=None):
    if tasks == None:
        tasks = tox21_tasks
    else:
        tasks = list(set(tasks))
        if not all(a in tox21_tasks for a in tasks):
            # Raise error
            pass
    data_list = []
    mols = {}
    for task in tasks:
        data_list.append(Tox(name = 'Tox21', label_name = task).get_data().rename(columns={"Y": task}))
        for _, row in data_list[-1].iterrows():
            if row['Drug_ID'] not in mols:
                mols[row['Drug_ID']] = row['Drug']
        data_list[-1] = data_list[-1].drop(['Drug'],axis=1)
    data_df = pd.DataFrame({'Drug_ID':mols.keys(),'Drug':mols.values()})
    for task_data in data_list:
        data_df = data_df.merge(task_data,how='outer',on='Drug_ID')
    return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

def get_Tox21_dataloader(tasks = None,
                        node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    if tasks == None:
        tasks = tox21_tasks

    return MolecularDataLoader(data=get_tox21(tasks=tasks),
                                task_names=tasks,
                                mol_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/tox21.bin',
                                mol_as_smiles=True,
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

#ToxCast

# toxcast_tasks = retrieve_label_name_list('Toxcast')

# def get_toxcast(tasks=None):
#     if tasks == None:
#         tasks = toxcast_tasks
#     else:
#         tasks = list(set(tasks))
#         if not all(a in toxcast_tasks for a in tasks):
#             # Raise error
#             pass
#     data_list = []
#     mols = {}
#     for task in tasks:
#         data_list.append(Tox(name = 'ToxCast', label_name = task).get_data().rename(columns={"Y": task}))
#         for _, row in data_list[-1].iterrows():
#             if row['Drug_ID'] not in mols:
#                 mols[row['Drug_ID']] = row['Drug']
#         data_list[-1] = data_list[-1].drop(['Drug'],axis=1)
#     data_df = pd.DataFrame({'Drug_ID':mols.keys(),'Drug':mols.values()})
#     for task_data in data_list:
#         data_df = data_df.merge(task_data,how='outer',on='Drug_ID')
#     return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

# def get_ToxCast_dataloader(tasks = None,
#                         node_featurizer=CanonicalAtomFeaturizer,
#                         edge_featurizer=CanonicalBondFeaturizer,
#                         batch_size=8,
#                         shuffle=True):

#     if tasks == None:
#         tasks = toxcast_tasks

#     return MolecularDataLoader(data=get_toxcast(tasks=tasks),
#                                 task_names=tasks,
#                                 mol_column='Smiles',
#                                 cache_file_path=CACHE_FOLDER+'/toxcast.bin',
#                                 mol_as_smiles=True,
#                                 node_featurizer=node_featurizer,
#                                 edge_featurizer=edge_featurizer,
#                                 batch_size=batch_size,
#                                 shuffle=shuffle)

def get_clintox():
    data_df = Tox(name = 'ClinTox').get_data()
    return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

def get_ClinTox_dataloader(node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    return MolecularDataLoader(data=get_clintox(),
                                task_names=['Y'],
                                mol_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/clintox.bin',
                                mol_as_smiles=True,
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

from tdc.single_pred import HTS
# HIV dataset

def get_hiv():
    data_df = HTS(name = 'HIV').get_data()
    return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

def get_HIV_dataloader(node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    return MolecularDataLoader(data=get_hiv(),
                                task_names=['Y'],
                                mol_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/hiv.bin',
                                mol_as_smiles=True,
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

# from tdc.single_pred import QM

# #QM7b
# qm7b_tasks = retrieve_label_name_list('QM7b')
# def get_qm7b(tasks=None):
#     if tasks == None:
#         tasks = qm7b_tasks
#     else:
#         tasks = list(set(tasks))
#         if not all(a in qm7b_tasks for a in tasks):
#             # Raise error
#             pass
#     data_list = []
#     mols = {}
#     for task in tasks:
#         data_list.append(QM(name = 'QM7b', label_name = task).get_data().rename(columns={"Y": task}))
#         for _, row in data_list[-1].iterrows():
#             if row['Drug_ID'] not in mols:
#                 mols[row['Drug_ID']] = row['Drug']
#         data_list[-1] = data_list[-1].drop(['Drug'],axis=1)
#     data_df = pd.DataFrame({'Drug_ID':mols.keys(),'Drug':mols.values()})
#     for task_data in data_list:
#         data_df = data_df.merge(task_data,how='outer',on='Drug_ID')
#     return data_df.rename(columns={'Drug_ID':'ID','Drug':'Coulomb_matrix'})

# def get_QM7b_dataloader(tasks = None,
#                         node_featurizer=CanonicalAtomFeaturizer,
#                         edge_featurizer=CanonicalBondFeaturizer,
#                         batch_size=8,
#                         shuffle=True):

#     if tasks == None:
#         tasks = qm7b_tasks

#     return MolecularDataLoader(data=get_qm7b(tasks=tasks),
#                                 task_names=tasks,
#                                 smile_column='Smiles',
#                                 cache_file_path=CACHE_FOLDER+'/qm7.bin',
#                                 node_featurizer=node_featurizer,
#                                 edge_featurizer=edge_featurizer,
#                                 batch_size=batch_size,
#                                 shuffle=shuffle)

# #QM8
# qm8_tasks = retrieve_label_name_list('QM8')
# def get_qm8(tasks=None):
#     if tasks == None:
#         tasks = qm8_tasks
#     else:
#         tasks = list(set(tasks))
#         if not all(a in qm8_tasks for a in tasks):
#             # Raise error
#             pass
#     data_list = []
#     mols = {}
#     for task in tasks:
#         data_list.append(QM(name = 'QM8', label_name = task).get_data().rename(columns={"Y": task}))
#         for _, row in data_list[-1].iterrows():
#             if row['Drug_ID'] not in mols:
#                 mols[row['Drug_ID']] = row['Drug']
#         data_list[-1] = data_list[-1].drop(['Drug'],axis=1)
#     data_df = pd.DataFrame({'Drug_ID':mols.keys(),'Drug':mols.values()})
#     for task_data in data_list:
#         data_df = data_df.merge(task_data,how='outer',on='Drug_ID')
#     return data_df.rename(columns={'Drug_ID':'ID','Drug':'Coulomb_matrix'})

# def get_QM8_dataloader(tasks = None,
#                         node_featurizer=CanonicalAtomFeaturizer,
#                         edge_featurizer=CanonicalBondFeaturizer,
#                         batch_size=8,
#                         shuffle=True):

#     if tasks == None:
#         tasks = qm8_tasks

#     return MolecularDataLoader(data=get_qm8(tasks=tasks),
#                                 task_names=tasks,
#                                 smile_column='Smiles',
#                                 cache_file_path=CACHE_FOLDER+'/qm8.bin',
#                                 node_featurizer=node_featurizer,
#                                 edge_featurizer=edge_featurizer,
#                                 batch_size=batch_size,
#                                 shuffle=shuffle)

#QM9
GDB9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
qm9_tasks = retrieve_label_name_list('QM9')
def get_qm9(tasks=None,get_atom_types=False):
    if tasks == None:
        tasks = qm9_tasks
    else:
        tasks = list(set(tasks))
        if not all(a in qm9_tasks for a in tasks):
            # Raise error
            pass
    dataset_file = os.path.join('data', "gdb9.sdf")
    if os.path.exists(dataset_file):
        print("Found local copy...")
    else:
        print("Downloading...")
        if not os.path.exists('data'):
            os.makedirs('data')
        dc.utils.data_utils.download_url(url=GDB9_URL, dest_dir='data')
        dc.utils.data_utils.untargz_file(
            os.path.join('data', "gdb9.tar.gz"), 'data')
    print('Loading...')
    labels = pd.read_csv('data/gdb9.sdf.csv',index_col=False)
    data = LoadSDF('data/gdb9.sdf',smilesName='Smiles',
                                     molColName='Molecule',includeFingerprints=False)
    data = data.rename(columns={'ID':'mol_id'})
    res_df = data.merge(labels, how='left', on='mol_id')
    if get_atom_types:
        atom_types = set()
        for mol in res_df['Molecule'].values:
            atom_types.update([atom.GetSymbol() for atom in mol.GetAtoms()])
        res_df.atom_types = atom_types
    return res_df

def get_QM9_dataloader(tasks = None,
                        node_featurizer=AtomTypeFeaturizer,
                        edge_featurizer=BondDistanceFeaturizer,
                        batch_size=8,
                        shuffle=True):

    if tasks == None:
        tasks = qm9_tasks

    df = get_qm9(tasks=tasks,get_atom_types=True)
    atom_types = df.atom_types

    return MolecularDataLoader(data=df,
                                task_names=tasks,
                                mol_column='Molecule',
                                cache_file_path=CACHE_FOLDER+'/qm9.bin',
                                mol_as_smiles=False,
                                node_featurizer=partial(node_featurizer,atom_types=atom_types),
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

