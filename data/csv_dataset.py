# Extension of DGL-life's MoleculeCSVDataset
import dgl.backend as F
import numpy as np
import os
import torch

from dgl.data.utils import save_graphs, load_graphs
from rdkit import Chem
from joblib import Parallel, delayed, cpu_count

# Obtain from https://github.com/awslabs/dgl-lifesci/blob/0149145f84a27aeca37f6594ca5793b1e935cf07/python/dgllife/utils/io.py
def pmap(pickleable_fn, data, n_jobs=None, verbose=1, **kwargs):
    """Parallel map using joblib.
    Parameters
    ----------
    pickleable_fn : callable
        Function to map over data.
    data : iterable
        Data over which we want to parallelize the function call.
    n_jobs : int, optional
        The maximum number of concurrently running jobs. By default, it is one less than
        the number of CPUs.
    verbose: int, optional
        The verbosity level. If nonzero, the function prints the progress messages.
        The frequency of the messages increases with the verbosity level. If above 10,
        it reports all iterations. If above 50, it sends the output to stdout.
    kwargs
        Additional arguments for :attr:`pickleable_fn`.
    Returns
    -------
    list
        The i-th element of the list corresponds to the output of applying
        :attr:`pickleable_fn` to :attr:`data[i]`.
    """
    if n_jobs is None:
        n_jobs = cpu_count() - 1

    return Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(pickleable_fn)(d, **kwargs) for d in data
    )

# Obtain and extend from https://github.com/awslabs/dgl-lifesci/blob/0149145f84a27aeca37f6594ca5793b1e935cf07/python/dgllife/data/csv_dataset.py
class MoleculeCSVDataset(object):
    """MoleculeCSVDataset

    This is a general class for loading molecular data from :class:`pandas.DataFrame`.

    In data pre-processing, we construct a binary mask indicating the existence of labels.

    All molecules are converted into DGLGraphs. After the first-time construction, the
    DGLGraphs can be saved for reloading so that we do not need to reconstruct them every time.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe including smiles and labels. Can be loaded by pandas.read_csv(file_path).
        One column includes smiles and some other columns include labels.
    to_graph_func: callable, Any -> DGLGraph
        A function turning a molecule representation into a DGLGraph.
    node_featurizer : None or callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    edge_featurizer : None or callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph.
    mol_column: str
        Column name for molecule representation in ``df``.
    cache_file_path: str
        Path to store the preprocessed DGLGraphs. For example, this can be ``'dglgraph.bin'``.
    mol_as_smiles: bool
        Whether the molecule representations are Smiles.
    task_names : list of str or None
        Columns in the data frame corresponding to real-valued labels. If None, we assume
        all columns except the smiles_column are labels. Default to None.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to False.
    log_every : bool
        Print a message every time ``log_every`` molecules are processed. It only comes
        into effect when :attr:`n_jobs` is greater than 1. Default to 1000.
    init_mask : bool
        Whether to initialize a binary mask indicating the existence of labels. Default to True.
    n_jobs : int
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.
    """
    def __init__(self, df, to_graph_func, node_featurizer, edge_featurizer, mol_column,
                 cache_file_path, mol_as_smiles, task_names=None, load=False, log_every=1000, init_mask=True,
                 n_jobs=1):
        self.df = df
        self.mol_as_smiles = mol_as_smiles
        self.mols = self.df[mol_column].tolist()
        if task_names is None:
            self.task_names = self.df.columns.drop([mol_column]).tolist()
        else:
            self.task_names = task_names
        self.n_tasks = len(self.task_names)
        self.cache_file_path = cache_file_path
        self._pre_process(to_graph_func, node_featurizer, edge_featurizer,
                          load, log_every, init_mask, n_jobs)

        # Only useful for binary classification tasks
        self._task_pos_weights = None

    def _pre_process(self, to_graph_func, node_featurizer,
                     edge_featurizer, load, log_every, init_mask, n_jobs=1):
        """Pre-process the dataset

        * Convert molecules from smiles format into DGLGraphs
          and featurize their atoms
        * Set missing labels to be 0 and use a binary masking
          matrix to mask them

        Parameters
        ----------
        smiles_to_graph : callable, SMILES -> DGLGraph
            Function for converting a SMILES (str) into a DGLGraph.
        node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
            Featurization for nodes like atoms in a molecule, which can be used to update
            ndata for a DGLGraph.
        edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
            Featurization for edges like bonds in a molecule, which can be used to update
            edata for a DGLGraph.
        load : bool
            Whether to load the previously pre-processed dataset or pre-process from scratch.
            ``load`` should be False when we want to try different graph construction and
            featurization methods and need to preprocess from scratch. Default to True.
        log_every : bool
            Print a message every time ``log_every`` molecules are processed. It only comes
            into effect when :attr:`n_jobs` is greater than 1.
        init_mask : bool
            Whether to initialize a binary mask indicating the existence of labels.
        n_jobs : int
            Degree of parallelism for pre processing. Default to 1.
        """
        if os.path.exists(self.cache_file_path) and load:
            # DGLGraphs have been constructed before, reload them
            print('Loading previously saved dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
            self.labels = label_dict['labels']
            if init_mask:
                self.mask = label_dict['mask']
            self.valid_ids = label_dict['valid_ids'].tolist()
        else:
            print('Processing dgl graphs from scratch...')
            if n_jobs > 1:
                self.graphs = pmap(to_graph_func,
                                   self.mols,
                                   node_featurizer=node_featurizer,
                                   edge_featurizer=edge_featurizer,
                                   n_jobs=n_jobs)
            else:
                self.graphs = []
                for i, s in enumerate(self.mols):
                    if (i + 1) % log_every == 0:
                        print('Processing molecule {:d}/{:d}'.format(i+1, len(self)))
                    self.graphs.append(to_graph_func(s, node_featurizer=node_featurizer,
                                                       edge_featurizer=edge_featurizer))

            # Keep only valid molecules
            self.valid_ids = []
            graphs = []
            for i, g in enumerate(self.graphs):
                if g is not None:
                    self.valid_ids.append(i)
                    graphs.append(g)
            self.graphs = graphs
            _label_values = self.df[self.task_names].values
            # np.nan_to_num will also turn inf into a very large number
            self.labels = F.zerocopy_from_numpy(
                np.nan_to_num(_label_values).astype(np.float32))[self.valid_ids]
            valid_ids = torch.tensor(self.valid_ids)
            if init_mask:
                self.mask = F.zerocopy_from_numpy(
                    (~np.isnan(_label_values)).astype(np.float32))[self.valid_ids]
                save_graphs(self.cache_file_path, self.graphs,
                            labels={'labels': self.labels, 'mask': self.mask,
                                    'valid_ids': valid_ids})
            else:
                self.mask = None
                save_graphs(self.cache_file_path, self.graphs,
                            labels={'labels': self.labels, 'valid_ids': valid_ids})

        self.mols = [self.mols[i] for i in self.valid_ids]
        if not self.mol_as_smiles:
            self.smiles = [Chem.MolToSmiles(self.mols[i]) for i in self.valid_ids]
        else:
            self.smiles = [x for x in self.mols]

    def __getitem__(self, item):
        """Get datapoint with index

        Parameters
        ----------
        item : int
            Datapoint index

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32 and shape (T)
            Labels of the datapoint for all tasks
        Tensor of dtype float32 and shape (T), optional
            Binary masks indicating the existence of labels for all tasks. This is only
            generated when ``init_mask`` is True in the initialization.
        """
        if self.mask is not None:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]

    def __len__(self):
        """Size for the dataset

        Returns
        -------
        int
            Size for the dataset
        """
        return len(self.mols)

    def task_pos_weights(self, indices):
        """Get weights for positive samples on each task

        This should only be used when all tasks are binary classification.

        It's quite common that the number of positive samples and the number of
        negative samples are significantly different for binary classification.
        To compensate for the class imbalance issue, we can weight each datapoint
        in loss computation.

        In particular, for each task we will set the weight of negative samples
        to be 1 and the weight of positive samples to be the number of negative
        samples divided by the number of positive samples.

        Parameters
        ----------
        indices : 1D LongTensor
            The function will compute the weights on the data subset specified by
            the indices, e.g. the indices for the training set.

        Returns
        -------
        Tensor of dtype float32 and shape (T)
            Weight of positive samples on all tasks
        """
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = F.sum(self.labels[indices], dim=0)
        num_indices = F.sum(self.mask[indices], dim=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]

        return task_pos_weights
