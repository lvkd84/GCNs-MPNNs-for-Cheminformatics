# Based on DGL-LifeSci's featurizer infrastructure

import numpy as np
import torch
from dgllife.utils import ConcatFeaturizer, BaseAtomFeaturizer, BaseBondFeaturizer, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from rdkit import Chem
from rdkit.Chem import AllChem
from dgl import backend as F
from functools import partial
from collections import defaultdict

########### ATOM Featurizing Functions ############
def atomTypeNum(atom, allowable_dic):
    return [allowable_dic[atom.GetSymbol()]]

################ ATOM Featurizers #################
class CombinedAtomFeaturizer(BaseAtomFeaturizer):
    pass

"""
DGL-LifeSci's Canonical Atom Featurizer. 
Including here for collecting and lookup purposes. 
"""
CanonicalAtomFeaturizer = CanonicalAtomFeaturizer

"""
Not to be confused with one-hot encoding of atoms.

Encode a unique number for each atom type.

To be used in models that use a lookup table for the input representation of an atom type: SchNet
"""
class AtomTypeFeaturizer(BaseAtomFeaturizer):
    def __init__(self,atom_data_field='atom_type',atom_types=None):

        if atom_types is None:
            atom_types = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        atom_dic = {atom_type:index for index, atom_type in enumerate(atom_types)}

        super(AtomTypeFeaturizer,self).__init__(
            featurizer_funcs={atom_data_field:partial(atomTypeNum,allowable_dic=atom_dic)})     

########### BOND Featurizing Functions ############


################ BOND Featurizers #################
class CombinedBondFeaturizer(BaseBondFeaturizer):
    pass

"""
DGL-LifeSci's Canonical Bond Featurizer. 
Including here for collecting and lookup purposes. 
"""
CanonicalBondFeaturizer = CanonicalBondFeaturizer

class BondDistanceFeaturizer:
    def __init__(self,bond_data_field='bond_dist',self_loop=False):
        self.bond_data_field = bond_data_field
        self._self_loop = self_loop

    # Adopt from DeepChem: https://github.com/deepchem/deepchem/blob/master/deepchem/feat/molecule_featurizers/coulomb_matrices.py
    def get_interatomic_distances(self,conf,bond):
        coord_i = conf.GetAtomPosition(bond.GetBeginAtomIdx()).__idiv__(0.52917721092)
        coord_j = conf.GetAtomPosition(bond.GetEndAtomIdx()).__idiv__(0.52917721092)
        return [coord_i.Distance(coord_j)]

    def __call__(self,mol):
        """Featurize all bonds in a molecule.
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.
        Returns
        -------
        dict
            For each function in self.featurizer_funcs with the key ``k``, store the computed
            feature under the key ``k``. Each feature is a tensor of dtype float32 and shape
            (N, M), where N is the number of atoms in the molecule.
        """
        num_bonds = mol.GetNumBonds()
        bond_features = defaultdict(list)

        num_confs = len(mol.GetConformers())
        if num_confs == 0:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if len(mol.GetConformers()) == 0: # Fail to calculate coords
            # Assign default dist to each bond
            for i in range(num_bonds):
                bond = mol.GetBondWithIdx(i)
                feat = [1.]
                bond_features[self.bond_data_field].extend([feat,feat.copy()])
        else:
            conf = mol.GetConformers()[0]
            # Compute features for each bond
            for i in range(num_bonds):
                bond = mol.GetBondWithIdx(i)
                feat = self.get_interatomic_distances(conf,bond)
                bond_features[self.bond_data_field].extend([feat,feat.copy()])

        # convert the feature to a float array
        processed_features = dict()
        if len(bond_features) > 0:
            feat = np.stack(bond_features[self.bond_data_field])
            processed_features[self.bond_data_field] = F.zerocopy_from_numpy(feat.astype(np.float32))

        if self._self_loop and num_bonds > 0:
            num_atoms = mol.GetNumAtoms()
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.cat([feats, torch.zeros(feats.shape[0], 1)], dim=1)
                self_loop_feats = torch.zeros(num_atoms, feats.shape[1])
                self_loop_feats[:, -1] = 1
                feats = torch.cat([feats, self_loop_feats], dim=0)
                processed_features[feat_name] = feats

        if self._self_loop and num_bonds == 0:
            num_atoms = mol.GetNumAtoms()
            toy_mol = Chem.MolFromSmiles('CO')
            processed_features = self(toy_mol)
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.zeros(num_atoms, feats.shape[1])
                feats[:, -1] = 1
                processed_features[feat_name] = feats

        return processed_features