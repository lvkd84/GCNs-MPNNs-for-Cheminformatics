import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

# Only the convolution
class DMPNNConv(nn.Module):
    
    def __init__(self, feats, weight_mtx):
        super(DMPNNConv, self).__init__()
        self.feats = feats
        self.w = weight_mtx

    def forward(self, g, efeat, initial_efeat):
        with g.local_scope():
            g.edata['h0'] = initial_efeat
            g.edata['h'] = efeat
            g.update_all(fn.copy_e('h','s'),
                         fn.sum('s','m'))
            g.apply_edges(lambda edges:{'h1':edges.src['m']-edges.data['h']})
            g.apply_edges(lambda edges:{'h2':edges.data['h0']+self.w(edges.data['h1'])})

            rst = g.edata.pop('h2')
            return rst

# Convolution + Others
class DMPNNGNN(nn.Module):

    def __init__(self, 
                 in_node_feats, 
                 in_edge_feats, 
                 out_node_feats, 
                 hidden_edge_feats, 
                 num_steps_passing=6):
        super(DMPNNGNN, self).__init__()

        self.num_steps_passing = num_steps_passing

        self.W_i = nn.Linear(in_node_feats+in_edge_feats,
                             hidden_edge_feats,
                             bias=False)

        self.W_m = nn.Linear(hidden_edge_feats,hidden_edge_feats,
                             bias=False)

        self.W_a = nn.Linear(in_node_feats+hidden_edge_feats,
                             out_node_feats,
                             bias=False)

        self.gnn = DMPNNConv(feats=hidden_edge_feats,
                             weight_mtx=self.W_m)


    def reset_parameters(self):
        self.gnn.reset_parameters()
    
    def forward(self, g, node_feats, edge_feats):
        g.ndata['x'] = node_feats
        g.edata['h'] = edge_feats
        g.apply_edges(lambda edges:{'h0':torch.cat((edges.src['x'],edges.data['h']),dim=1)})
        g.ndata.pop('x')
        g.edata.pop('h')
        initial_efeats = F.relu(self.W_i(g.edata.pop('h0')))
        edge_feats = initial_efeats.clone().detach()
        for _ in range(self.num_steps_passing):
            edge_feats = F.relu(self.gnn(g,edge_feats,initial_efeats))
        g.edata['h'] = edge_feats
        g.update_all(fn.copy_e('h','m'),
                     fn.sum('m','x'))
        output_node_feats = g.ndata.pop('x')
        final_node_feats = F.relu(self.W_a(torch.cat((node_feats,output_node_feats),dim=1)))
        return final_node_feats

from dgl.nn import SumPooling
# Convolutions + Readouts + FFNN
class DMPNNPredictor(nn.Module):
    
    def __init__(self,
                 in_node_feats, 
                 in_edge_feats, 
                 out_node_feats=64,
                 hidden_edge_feats=128,
                 num_tasks=1,
                 num_steps_passing=6):
        super(DMPNNPredictor, self).__init__()

        self.gnn = DMPNNGNN(in_node_feats, 
                            in_edge_feats, 
                            out_node_feats, 
                            hidden_edge_feats,
                            num_steps_passing)

        self.readout = SumPooling()

        self.predict = nn.Sequential(
            nn.Linear(out_node_feats,out_node_feats),
            nn.ReLU(),
            nn.Linear(out_node_feats,num_tasks)
        )
    
    def forward(self, g, node_feats, edge_feats):
        node_feats = self.gnn(g, node_feats, edge_feats)
        g_feats = self.readout(g, node_feats)
        return self.predict(g_feats)