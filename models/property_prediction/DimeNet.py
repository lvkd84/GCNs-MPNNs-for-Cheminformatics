# Obtained from https://github.com/xnuohz/DimeNet-dgl

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sympy as sym

import dgl
import dgl.function as fn

from dime_layers import *
from dime_utils import *

# Interaction layer
class DimeConv(nn.Module):
    def __init__(self,
                 emb_size,
                 num_radial,
                 num_spherical,
                 num_bilinear,
                 num_before_skip,
                 num_after_skip,
                 activation=None):
        super(DimeConv, self).__init__()

        self.activation = activation
        # Transformations of Bessel and spherical basis representations
        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.dense_sbf = nn.Linear(num_radial * num_spherical, num_bilinear, bias=False)
        # Dense transformations of input messages
        self.dense_ji = nn.Linear(emb_size, emb_size)
        self.dense_kj = nn.Linear(emb_size, emb_size)
        # Bilinear layer
        bilin_initializer = torch.empty((emb_size, num_bilinear, emb_size)).normal_(mean=0, std=2 / emb_size)
        self.W_bilin = nn.Parameter(bilin_initializer)
        # Residual layers before skip connection
        self.layers_before_skip = nn.ModuleList([
            ResidualLayer(emb_size, activation=activation) for _ in range(num_before_skip)
        ])
        self.final_before_skip = nn.Linear(emb_size, emb_size)
        # Residual layers after skip connection
        self.layers_after_skip = nn.ModuleList([
            ResidualLayer(emb_size, activation=activation) for _ in range(num_after_skip)
        ])

        self.reset_params()
    
    def reset_params(self):
        GlorotOrthogonal(self.dense_rbf.weight)
        GlorotOrthogonal(self.dense_sbf.weight)
        GlorotOrthogonal(self.dense_ji.weight)
        GlorotOrthogonal(self.dense_kj.weight)
        GlorotOrthogonal(self.final_before_skip.weight)

    def edge_transfer(self, edges):
        # Transform from Bessel basis to dence vector
        rbf = self.dense_rbf(edges.data['rbf'])
        # Initial transformation
        x_ji = self.dense_ji(edges.data['m'])
        x_kj = self.dense_kj(edges.data['m'])
        if self.activation is not None:
            x_ji = self.activation(x_ji)
            x_kj = self.activation(x_kj)

        # w: W * e_RBF \bigodot \sigma(W * m + b)
        return {'x_kj': x_kj * rbf, 'x_ji': x_ji}

    def msg_func(self, edges):
        sbf = self.dense_sbf(edges.data['sbf'])
        # Apply bilinear layer to interactions and basis function activation
        # [None, 8] * [128, 8, 128] * [None, 128] -> [None, 128]
        x_kj = torch.einsum("wj,wl,ijl->wi", sbf, edges.src['x_kj'], self.W_bilin)
        return {'x_kj': x_kj}

    def forward(self, g, l_g):
        g.apply_edges(self.edge_transfer)
        
        # nodes correspond to edges and edges correspond to nodes in the original graphs
        # node: d, rbf, o, rbf_env, x_kj, x_ji
        for k, v in g.edata.items():
            l_g.ndata[k] = v

        l_g.update_all(self.msg_func, fn.sum('x_kj', 'm_update'))

        for k, v in l_g.ndata.items():
            g.edata[k] = v

        # Transformations before skip connection
        g.edata['m_update'] = g.edata['m_update'] + g.edata['x_ji']
        for layer in self.layers_before_skip:
            g.edata['m_update'] = layer(g.edata['m_update'])
        g.edata['m_update'] = self.final_before_skip(g.edata['m_update'])
        if self.activation is not None:
            g.edata['m_update'] = self.activation(g.edata['m_update'])

        # Skip connection
        g.edata['m'] = g.edata['m'] + g.edata['m_update']

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            g.edata['m'] = layer(g.edata['m'])

        return g

class DimePredictor(nn.Module):
    """
    DimeNet model.
    Parameters
    ----------
    emb_size
        Embedding size used throughout the model
    num_blocks
        Number of building blocks to be stacked
    num_bilinear
        Third dimension of the bilinear layer tensor
    num_spherical
        Number of spherical harmonics
    num_radial
        Number of radial basis functions
    cutoff
        Cutoff distance for interatomic interactions
    envelope_exponent
        Shape of the smooth cutoff
    num_before_skip
        Number of residual layers in interaction block before skip connection
    num_after_skip
        Number of residual layers in interaction block after skip connection
    num_dense_output
        Number of dense layers for the output blocks
    num_targets
        Number of targets to predict
    activation
        Activation function
    output_init
        Initial function in output block
    """
    def __init__(self,
                 emb_size,
                 num_blocks,
                 num_bilinear,
                 num_spherical,
                 num_radial,
                 cutoff=5.0,
                 envelope_exponent=5,
                 num_before_skip=1,
                 num_after_skip=2,
                 num_dense_output=3,
                 num_targets=12,
                 activation=swish,
                 output_init=nn.init.zeros_):
        super(DimePredictor, self).__init__()

        self.num_blocks = num_blocks
        self.num_radial = num_radial

        # cosine basis function expansion layer
        self.rbf_layer = RBFLayer(num_radial=num_radial,
                                          cutoff=cutoff,
                                          envelope_exponent=envelope_exponent)

        self.sbf_layer = SBFLayer(num_spherical=num_spherical,
                                             num_radial=num_radial,
                                             cutoff=cutoff,
                                             envelope_exponent=envelope_exponent)
        
        # embedding block
        self.emb_block = EmbeddingLayer(emb_size=emb_size,
                                        num_radial=num_radial,
                                        bessel_funcs=self.sbf_layer.get_bessel_funcs(),
                                        cutoff=cutoff,
                                        envelope_exponent=envelope_exponent,
                                        activation=activation)
        
        # output block
        self.output_blocks = nn.ModuleList({
            OutputLayer(emb_size=emb_size,
                        num_radial=num_radial,
                        num_dense=num_dense_output,
                        num_targets=num_targets,
                        activation=activation,
                        output_init=output_init) for _ in range(num_blocks + 1)
        })

        # interaction block
        self.interaction_blocks = nn.ModuleList({
            InteractionLayer(emb_size=emb_size,
                             num_radial=num_radial,
                             num_spherical=num_spherical,
                             num_bilinear=num_bilinear,
                             num_before_skip=num_before_skip,
                             num_after_skip=num_after_skip,
                             activation=activation) for _ in range(num_blocks)
        })
    
    def edge_init(self, edges):
        # Calculate angles k -> j -> i
        R1, R2 = edges.src['o'], edges.dst['o']
        x = torch.sum(R1 * R2, dim=-1)
        y = torch.cross(R1, R2)
        y = torch.norm(y, dim=-1)
        angle = torch.atan2(y, x)
        # Transform via angles
        cbf = [f(angle) for f in self.sbf_layer.get_sph_funcs()]
        cbf = torch.stack(cbf, dim=1)  # [None, 7]
        cbf = cbf.repeat_interleave(self.num_radial, dim=1)  # [None, 42]
        sbf = edges.src['rbf_env'] * cbf  # [None, 42]
        return {'sbf': sbf}
    
    def forward(self, g):
        # add rbf features for each edge in one batch graph, [num_radial,]
        g = self.rbf_layer(g)
        # Embedding block
        g = self.emb_block(g)
        # Output block
        P = self.output_blocks[0](g)  # [batch_size, num_targets]
        # Prepare sbf feature before the following blocks
        l_g = dgl.line_graph(g, backtracking=False)
        for k, v in g.edata.items():
            l_g.ndata[k] = v

        l_g.apply_edges(self.edge_init)
        # Interaction blocks
        for i in range(self.num_blocks):
            g = self.interaction_blocks[i](g, l_g)
            P += self.output_blocks[i + 1](g)
        
        return P