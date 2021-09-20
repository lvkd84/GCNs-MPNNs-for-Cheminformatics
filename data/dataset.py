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
                data_path, 
                task_names,
                smile_column,
                cache_file_path,
                node_featurizer=CanonicalAtomFeaturizer,
                edge_featurizer=CanonicalBondFeaturizer,
                batch_size=8,
                shuffle=True):

        self.df = pd.read_csv(data_path)
        self.tasks = task_names       
        self.dataset = MoleculeCSVDataset(df=self.df,
                        smiles_to_graph=smiles_to_bigraph,
                        cache_file_path=cache_file_path,
                        node_featurizer=node_featurizer(atom_data_field='x'),
                        edge_featurizer=edge_featurizer(bond_data_field='edge_attr'),
                        smiles_column=smile_column,
                        task_names=task_names)
        self.dataloader = DataLoader(self.dataset,
                                    collate_fn=collate_molgraphs,
                                    batch_size=batch_size,
                                    shuffle=shuffle)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        try:
            return next(self.dataloader)
        except StopIteration:
            raise StopIteration

    def get_num_tasks(self):
        return len(self.tasks)

import os
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
CACHE_FOLDER = os.path.abspath(os.getcwd())

LIPO_PATH = CURRENT_FOLDER + '/datasets/' + 'Lipophilicity.csv'
def get_LIPO_dataloader(node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    return MolecularDataLoader(data_path=LIPO_PATH,
                                task_names='exp',
                                smile_column='smiles',
                                cache_file_path=CACHE_FOLDER+'/LIPO.bin',
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                batch_size=batch_size,
                                shuffle=shuffle)

SIDER_PATH = CURRENT_FOLDER + '/datasets/' + 'sider.csv'
sider_tasks = ["Hepatobiliary disorders",
                "Metabolism and nutrition disorders",
                "Product issues",
                "Eye disorders",
                "Investigations",
                "Musculoskeletal and connective tissue disorders",
                "Gastrointestinal disorders",
                "Social circumstances",
                "Immune system disorders",
                "Reproductive system and breast disorders",
                "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                "General disorders and administration site conditions",
                "Endocrine disorders",
                "Surgical and medical procedures",
                "Vascular disorders",
                "Blood and lymphatic system disorders",
                "Skin and subcutaneous tissue disorders",
                "Congenital, familial and genetic disorders",
                "Infections and infestations",
                "Respiratory, thoracic and mediastinal disorders",
                "Psychiatric disorders",
                "Renal and urinary disorders",
                "Pregnancy, puerperium and perinatal conditions",
                "Ear and labyrinth disorders",
                "Cardiac disorders",
                "Nervous system disorders",
                "Injury, poisoning and procedural complications"]
def get_SIDER_dataloader(node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    return MolecularDataLoader(data_path=SIDER_PATH,
                            task_names=sider_tasks,
                            smile_column='smiles',
                            cache_file_path=CACHE_FOLDER+'/SIDER.bin',
                            node_featurizer=node_featurizer,
                            edge_featurizer=edge_featurizer,
                            batch_size=batch_size,
                            shuffle=shuffle)

BBBP_PATH = CURRENT_FOLDER + '/datasets/' + 'BBBP.csv'
def get_BBBP_dataloader(node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):
    
    return MolecularDataLoader(data_path=BBBP_PATH,
                            task_names='p_np',
                            smile_column='smiles',
                            cache_file_path=CACHE_FOLDER+'/BBBP.bin',
                            node_featurizer=node_featurizer,
                            edge_featurizer=edge_featurizer,
                            batch_size=batch_size,
                            shuffle=shuffle)

def get_HIV_dataset():
    pass

MUV_PATH = CURRENT_FOLDER + '/datasets/' + 'muv.csv'
muv_tasks = ['MUV-466',
            'MUV-548',
            'MUV-600',
            'MUV-644',
            'MUV-652',
            'MUV-689',
            'MUV-692',
            'MUV-712',
            'MUV-713',
            'MUV-733',
            'MUV-737',
            'MUV-810',
            'MUV-832',
            'MUV-846',
            'MUV-852',
            'MUV-858',
            'MUV-859']
def get_MUV_dataloader(node_featurizer=CanonicalAtomFeaturizer,
                        edge_featurizer=CanonicalBondFeaturizer,
                        batch_size=8,
                        shuffle=True):

    return MolecularDataLoader(data_path=MUV_PATH,
                            task_names=muv_tasks,
                            smile_column='smiles',
                            cache_file_path=CACHE_FOLDER+'/MUV.bin',
                            node_featurizer=node_featurizer,
                            edge_featurizer=edge_featurizer,
                            batch_size=batch_size,
                            shuffle=shuffle)

def get_ESOL_dataset():
    pass

def get_TOX21_dataset():
    pass

