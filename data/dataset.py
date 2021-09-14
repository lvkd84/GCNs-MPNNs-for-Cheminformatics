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
                node_featurizer=CanonicalAtomFeaturizer,
                edge_featurizer=CanonicalBondFeaturizer,
                batch_size=8,
                shuffle=True):

        df = pd.read_csv(data_path)       
        self.dataset = MoleculeCSVDataset(df=df,
                        smiles_to_graph=smiles_to_bigraph,
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

def get_LIPO_dataset():
    pass

def get_SIDER_dataset():
    pass

def get_BBBP_dataset():
    pass

def get_HIV_dataset():
    pass

def get_MUV_dataset():
    pass

def get_ESOL_dataset():
    pass

def get_TOX21_dataset():
    pass

