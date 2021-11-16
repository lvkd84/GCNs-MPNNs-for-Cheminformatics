from typing import Iterable
import pandas as pd
import torch
from torch.utils.data import DataLoader
import dgl
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks


class MolecularDataLoader(Iterable):
    """
    Data loader for molecular data.
    Parameters
    ----------
    data_path
        Path to the data file. File must be csv.
    task_names
        Names of label columns. Each column corresponds to a task.
    smile_column
        Name of the column that contains the SMILE strings.
    message_hidden_feats
        Number of hidden features.
    node_featurizer
        Callable node featurizing function.
    edge_featurizer
        Callable edge featurizing function.
    """    
    def __init__(self, 
                data, 
                task_names,
                smile_column,
                cache_file_path,
                node_featurizer=CanonicalAtomFeaturizer,
                edge_featurizer=CanonicalBondFeaturizer,
                batch_size=8,
                shuffle=True):

        self.df = data
        self.tasks = task_names       
        self.dataset = MoleculeCSVDataset(df=self.df,
                        smiles_to_graph=smiles_to_bigraph,
                        cache_file_path=cache_file_path,
                        node_featurizer=node_featurizer(atom_data_field='x'),
                        edge_featurizer=edge_featurizer(bond_data_field='edge_attr'),
                        smiles_column=smile_column,
                        task_names=task_names)

        _, graph, labels, _ = self.dataset[0]
        self.num_node_attrs = graph.ndata['x'].shape[1]
        self.num_edge_attrs = graph.edata['edge_attr'].shape[1]
        self.num_tasks = labels.shape[0]

        self.dataloader = DataLoader(self.dataset,
                                    collate_fn=collate_molgraphs,
                                    batch_size=batch_size,
                                    shuffle=shuffle)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataloader)

    def get_num_tasks(self):
        return len(self.tasks)

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
                                smile_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/Caco2.bin',
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
                                smile_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/Caco2.bin',
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
                                smile_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/Pgb.bin',
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
                                smile_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/LIPO.bin',
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
                                smile_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/AqSol.bin',
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
                                smile_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/FreeSolv.bin',
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
                                smile_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/PPBR.bin',
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
                                smile_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/VDss.bin',
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
                                smile_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/tox21.bin',
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

# ToxCast

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
#                                 smile_column='Smiles',
#                                 cache_file_path=CACHE_FOLDER+'/toxcast.bin',
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
                                smile_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/clintox.bin',
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
                                smile_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/hiv.bin',
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

from tdc.single_pred import QM

#QM7b
qm7b_tasks = retrieve_label_name_list('QM7b')
def get_qm7b(tasks=None):
    if tasks == None:
        tasks = qm7b_tasks
    else:
        tasks = list(set(tasks))
        if not all(a in qm7b_tasks for a in tasks):
            # Raise error
            pass
    data_list = []
    mols = {}
    for task in tasks:
        data_list.append(QM(name = 'QM7b', label_name = task).get_data().rename(columns={"Y": task}))
        for _, row in data_list[-1].iterrows():
            if row['Drug_ID'] not in mols:
                mols[row['Drug_ID']] = row['Drug']
        data_list[-1] = data_list[-1].drop(['Drug'],axis=1)
    data_df = pd.DataFrame({'Drug_ID':mols.keys(),'Drug':mols.values()})
    for task_data in data_list:
        data_df = data_df.merge(task_data,how='outer',on='Drug_ID')
    return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

def get_QM7b_dataloader(tasks = None,
                        node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    if tasks == None:
        tasks = qm7b_tasks

    return MolecularDataLoader(data=get_qm7b(tasks=tasks),
                                task_names=tasks,
                                smile_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/qm7.bin',
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

#QM8
qm8_tasks = retrieve_label_name_list('QM8')
def get_qm8(tasks=None):
    if tasks == None:
        tasks = qm8_tasks
    else:
        tasks = list(set(tasks))
        if not all(a in qm8_tasks for a in tasks):
            # Raise error
            pass
    data_list = []
    mols = {}
    for task in tasks:
        data_list.append(QM(name = 'QM8', label_name = task).get_data().rename(columns={"Y": task}))
        for _, row in data_list[-1].iterrows():
            if row['Drug_ID'] not in mols:
                mols[row['Drug_ID']] = row['Drug']
        data_list[-1] = data_list[-1].drop(['Drug'],axis=1)
    data_df = pd.DataFrame({'Drug_ID':mols.keys(),'Drug':mols.values()})
    for task_data in data_list:
        data_df = data_df.merge(task_data,how='outer',on='Drug_ID')
    return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

def get_QM8_dataloader(tasks = None,
                        node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    if tasks == None:
        tasks = qm8_tasks

    return MolecularDataLoader(data=get_qm8(tasks=tasks),
                                task_names=tasks,
                                smile_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/qm8.bin',
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

#QM9
qm9_tasks = retrieve_label_name_list('QM8')
def get_qm9(tasks=None):
    if tasks == None:
        tasks = qm9_tasks
    else:
        tasks = list(set(tasks))
        if not all(a in qm9_tasks for a in tasks):
            # Raise error
            pass
    data_list = []
    mols = {}
    for task in tasks:
        data_list.append(QM(name = 'QM9', label_name = task).get_data().rename(columns={"Y": task}))
        for _, row in data_list[-1].iterrows():
            if row['Drug_ID'] not in mols:
                mols[row['Drug_ID']] = row['Drug']
        data_list[-1] = data_list[-1].drop(['Drug'],axis=1)
    data_df = pd.DataFrame({'Drug_ID':mols.keys(),'Drug':mols.values()})
    for task_data in data_list:
        data_df = data_df.merge(task_data,how='outer',on='Drug_ID')
    return data_df.rename(columns={'Drug_ID':'ID','Drug':'Smiles'})

def get_QM9_dataloader(tasks = None,
                        node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    if tasks == None:
        tasks = qm9_tasks

    return MolecularDataLoader(data=get_qm9(tasks=tasks),
                                task_names=tasks,
                                smile_column='Smiles',
                                cache_file_path=CACHE_FOLDER+'/qm9.bin',
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

