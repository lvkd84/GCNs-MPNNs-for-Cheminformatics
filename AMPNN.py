import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import NNConv
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
import dgl.function as fn

class AMPNNConv(nn.Module):

    def __init__(self,in_feats,out_feats,msg_fn,attn_fn):
        super(AMPNNConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.msg_fn = msg_fn
        self.attn_fn = attn_fn

    def forward(self, g, feat, efeat):
        with g.local_scope():
            g.ndata['h'] = feat.unsqueeze(-1)

            g.edata['w_m'] = self.msg_fn(efeat).view(-1,self._in_feats,self._out_feats)
            g.edata['w_a'] = self.attn_fn(efeat).view(-1,self._in_feats,self._out_feats)

            g.apply_edges(lambda edges:{'e1':edges.data['w_m']*edges.src['h']})
            g.apply_edges(lambda edges:{'e2':edges.data['w_a']*edges.src['h']})

            e2 = g.edata.pop('e2')
            g.edata['attn'] = edge_softmax(g,e2)

            g.apply_edges(lambda edges:{'m':edges.data['e1']*edges.data['attn']})

            g.update_all(fn.copy_e('m','a'),
                         fn.sum('a','ft'))

            rst = g.dstdata['ft'].sum(dim=1)
            return rst

class AMPNNGNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=64,
                 edge_hidden_feats=128, num_step_message_passing=6):
        super(AMPNNGNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),
            nn.ReLU()
        )

        self.num_step_message_passing = num_step_message_passing

        message_nn = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )

        attention_nn = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )

        self.gnn_layer = AMPNNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            msg_fn=message_nn,
            attn_fn=attention_nn
        )

        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.project_node_feats(node_feats) # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)           # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            # print(node_feats.shape)
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)

        return node_feats


from dgl.nn.pytorch import GlobalAttentionPooling

# TODO: Check dimensions of layers
class AMPNNPredictor(nn.Module):

    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6):
        super(AMPNNPredictor, self).__init__()

        self.gnn = AMPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        # TODO: GG-NN Readout
        # self.readout = Set2Set(input_dim=node_out_feats,
        #                        n_iters=num_step_set2set,
        #                        n_layers=num_layer_set2set)

        self.gate_nn = nn.Sequential(
            nn.Linear(node_out_feats,1),
            nn.ReLU()
        )
        self.feat_nn = nn.Sequential(
            nn.Linear(node_out_feats,node_out_feats),
            nn.ReLU()
        )
        self.readout = GlobalAttentionPooling(self.gate_nn,self.feat_nn)

        self.predict = nn.Sequential(
            nn.Linear(node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Linear(node_out_feats, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)