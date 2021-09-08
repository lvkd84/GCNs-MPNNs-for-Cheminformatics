import torch
import torch.nn as nn
import torch.nn.functional as F

# from dgl.nn.functional import edge_softmax
import dgl.function as fn

# Only the convolution
class EMNNConv(nn.Module):
    
    def __init__(self, feats, msg_fn, attn_fn):
        super(EMNNConv, self).__init__()
        self.feats = feats
        self.msg_fn = msg_fn
        self.attn_fn = attn_fn

    def forward(self, g, efeat, initial_efeats):
        with g.local_scope():

            g.edata['w_m'] = self.msg_fn(efeat).view(-1,self.feats,self.feats)
            g.edata['w_a'] = self.attn_fn(efeat).view(-1,self.feats,self.feats)

            g.edata['h'] = efeat.unsqueeze(-1)
            g.edata['initial_h'] = initial_efeats.unsqueeze(-1)

            g.apply_edges(lambda edges:{'e1':edges.data['w_m']*edges.data['h']})
            g.apply_edges(lambda edges:{'e2':edges.data['w_a']*edges.data['h']})
            e2 = g.edata.pop('e2')
            g.edata['exp_e2'] = torch.exp(e2)

            g.apply_edges(lambda edges:{'initial_e1':edges.data['w_m']*edges.data['initial_h']})
            g.apply_edges(lambda edges:{'initial_e2':edges.data['w_a']*edges.data['initial_h']})
            initial_e2 = g.edata.pop('initial_e2')
            g.edata['exp_initial_e2'] = torch.exp(initial_e2)

            g.update_all(fn.copy_e('exp_e2','a'),
                         fn.sum('a','sum_exp'))
            
            g.apply_edges(lambda edges:{'sum_exp':edges.src['sum_exp']
                                        -edges.data['exp_e2']
                                        +edges.data['exp_initial_e2']})      
            g.apply_edges(lambda edges:{'h1':edges.data['exp_e2']*edges.data['e1']})
            g.apply_edges(lambda edges:{'initial_h1':edges.data['exp_initial_e2']
                                        *edges.data['initial_e1']})
            g.update_all(fn.copy_e('h1','b'),
                         fn.sum('b','m'))
            g.apply_edges(lambda edges:{'h2':(edges.src['m']-edges.data['h1']
                                              +edges.data['initial_h1'])/edges.data['sum_exp']})

            rst = g.edata['h2'].sum(dim=1)
            return rst

# Convolution + Others
class EMNNGNN(nn.Module):

    def __init__(self, 
                 in_node_feats, 
                 in_edge_feats, 
                 out_node_feats, 
                 hidden_edge_feats, 
                 num_steps_passing=6):
        super(EMNNGNN, self).__init__()

        self.num_steps_passing = num_steps_passing

        self.W_i = nn.Linear(2*in_node_feats+in_edge_feats,
                             hidden_edge_feats,
                             bias=False)
        
        message_nn = nn.Sequential(
            nn.Linear(hidden_edge_feats, hidden_edge_feats),
            nn.ReLU(),
            nn.Linear(hidden_edge_feats, hidden_edge_feats * hidden_edge_feats)
        )

        attention_nn = nn.Sequential(
            nn.Linear(hidden_edge_feats, hidden_edge_feats),
            nn.ReLU(),
            nn.Linear(hidden_edge_feats, hidden_edge_feats * hidden_edge_feats)
        )
        
        self.gnn = EMNNConv(feats=hidden_edge_feats,
                            msg_fn=message_nn,
                            attn_fn=attention_nn)
        
        self.gru = nn.GRU(hidden_edge_feats, hidden_edge_feats)


    def reset_parameters(self):
        self.gnn.reset_parameters()
    
    def forward(self, g, node_feats, edge_feats):
        g.ndata['x'] = node_feats
        g.edata['h'] = edge_feats
        g.apply_edges(lambda edges:{'h0':torch.cat((edges.src['x'],
                                                    edges.dst['x'],
                                                    edges.data['h'],),dim=1)})
        g.ndata.pop('x')
        g.edata.pop('h')
        initial_efeats = F.relu(self.W_i(g.edata.pop('h0')))
        edge_feats = initial_efeats.clone().detach()
        hidden_efeats = edge_feats.unsqueeze(0)
        for _ in range(self.num_steps_passing):
            edge_feats = F.relu(self.gnn(g,edge_feats,initial_efeats))
            edge_feats, hidden_efeats = self.gru(edge_feats.unsqueeze(0),hidden_efeats)
            edge_feats = edge_feats.squeeze(0)
        g.edata['h'] = edge_feats
        g.update_all(fn.copy_e('h','m'),
                     fn.sum('m','x'))
        output_node_feats = g.ndata.pop('x')
        return output_node_feats

from dgl.nn import GlobalAttentionPooling
# Convolutions + Readouts + FFNN
class EMNNPredictor(nn.Module):
    
    def __init__(self,
                 in_node_feats, 
                 in_edge_feats, 
                 out_node_feats=64,
                 hidden_edge_feats=128,
                 num_tasks=1,
                 num_steps_passing=6):
        super(EMNNPredictor, self).__init__()

        self.gnn = EMNNGNN(in_node_feats, 
                            in_edge_feats, 
                            out_node_feats, 
                            hidden_edge_feats,
                            num_steps_passing)

        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_edge_feats,1),
            nn.ReLU()
        )
        self.feat_nn = nn.Sequential(
            nn.Linear(hidden_edge_feats,out_node_feats),
            nn.ReLU()
        )
        self.readout = GlobalAttentionPooling(self.gate_nn,self.feat_nn)

        self.predict = nn.Sequential(
            nn.Linear(out_node_feats,out_node_feats),
            nn.ReLU(),
            nn.Linear(out_node_feats,num_tasks)
        )
    
    def forward(self, g, node_feats, edge_feats):
        node_feats = self.gnn(g, node_feats, edge_feats)
        g_feats = self.readout(g, node_feats)
        return self.predict(g_feats)