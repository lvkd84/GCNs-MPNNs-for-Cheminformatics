import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

from dgl import batch

# Only the convolution
class SAMPNConv(nn.Module):
    
    def __init__(self, feats, input_nn, hidden_nn):
        super(SAMPNConv, self).__init__()
        self.feats = feats
        self.input_nn = input_nn
        self.hidden_nn = hidden_nn

    def reset_parameters(self):
        self.input_nn.reset_parameters()
        self.hidden_nn.reset_parameters()

    def forward(self, g, node_feats, edge_feats, messages):
        with g.local_scope():
            g.edata['pre_mess'] = messages
            g.edata['xy'] = edge_feats
            g.ndata['x'] = node_feats

            g.update_all(fn.copy_e('pre_mess','s'),
                         fn.sum('s','_neighbor_sum'))
            
            g.apply_edges(lambda edges:{'neighbor_sum':edges.src['_neighbor_sum']
                                        -edges.data['pre_mess']})

            g.apply_edges(lambda edges:{'mess':self.input_nn(torch.cat((edges.src['x'],
                                                                        edges.data['xy']),
                                                                       dim=1))
                                               +self.hidden_nn(edges.data['neighbor_sum'])})
            rst = g.edata.pop('mess')
            return rst

# Convolution + Others
class SAMPNGNN(nn.Module):

    def __init__(self, 
                 node_in_feats,
                 node_out_feats,
                 edge_in_feats,
                 message_hidden_feats,
                 num_message_passing=6,
                 shared_message_passing_weights=True):
        super(SAMPNGNN, self).__init__()

        self.num_message_passing = num_message_passing

        self.shared_message_passing_weights = shared_message_passing_weights

        if self.shared_message_passing_weights:
            self.W_in = nn.Linear(node_in_feats+edge_in_feats,message_hidden_feats)

            self.W_h = nn.Linear(message_hidden_feats,message_hidden_feats)

            self.gnn_layer = SAMPNConv(message_hidden_feats,self.W_in,self.W_h)
        else:
            self.gnn_layers = nn.ModuleList()

            for _ in range(num_message_passing):
                W_in = nn.Linear(node_in_feats+edge_in_feats,message_hidden_feats)

                W_h = nn.Linear(message_hidden_feats,message_hidden_feats)

                gnn_layer = SAMPNConv(message_hidden_feats,W_in,W_h)

                self.gnn_layers.append(gnn_layer)

        self.W_ah = nn.Linear(node_in_feats,message_hidden_feats)

        self.W_o = nn.Linear(message_hidden_feats,node_out_feats)

    def reset_parameters(self):
        self.W_ah.reset_parameters()
        self.W_o.reset_parameters()
        if self.shared_message_passing_weights:
            self.gnn_layer.reset_parameters()
        else:
            for layer in self.gnn_layers:
                layer.reset_parameters()
    
    def forward(self, g, node_feats, edge_feats):
        g.ndata['node_feats'] = node_feats
        g.apply_edges(lambda edges:{'src_feats':edges.src['node_feats']})
        src_feats = g.edata.pop('src_feats')
        g.ndata.pop('node_feats')
        if self.shared_message_passing_weights:
            messages = torch.relu(self.W_in(torch.cat((src_feats,edge_feats),dim=1)))
            for _ in range(self.num_message_passing):
                messages = self.gnn_layer(g, node_feats, edge_feats, messages)
        else:
            messages = torch.relu(self.gnn_layers[0].input_nn(torch.cat((src_feats,edge_feats),dim=1)))
            for _ in range(self.num_message_passing):
                messages = self.gnn_layers[_](g, node_feats, edge_feats, messages)
        g.edata['message'] = messages
        g.update_all(fn.copy_e('message','s'),
                     fn.sum('s','neighbor_sum'))
        neighbor_sum = g.ndata.pop('neighbor_sum')
        node_hidden_feats = torch.relu(self.W_o(self.W_ah(node_feats) + neighbor_sum))
        W_att = torch.softmax(torch.mm(node_hidden_feats,torch.transpose(node_hidden_feats,0,1)),dim=0)
        EG = torch.mm(W_att,node_hidden_feats)
        return EG+node_hidden_feats

from dgl.nn import AvgPooling
# Convolutions + Readouts + FFNN
class SAMPNPredictor(nn.Module):
    """
    SAMPN model.
    Parameters
    ----------
    edge_in_feats
        Number of input edge features
    node_in_feats
        Number of input node features
    node_out_feats
        Number of output node features
    message_hidden_feats
        Number of hidden features
    num_tasks
        Number of prediction tasks
    num_message_passing
        Number of message passing layers
    drop_out_rate
        The drop-out rate at the fully-connected layers
    shared_message_passing_weights
        Whether the weights are shared among the message passing layers
    """    
    def __init__(self,
                 edge_in_feats,
                 node_in_feats,
                 node_out_feats=64,
                 message_hidden_feats=128,
                 num_tasks=1,
                 num_message_passing=6,
                 drop_out_rate=0,
                 shared_message_passing_weights=True):
        super(SAMPNPredictor, self).__init__()

        self.gnn = SAMPNGNN(node_in_feats=node_in_feats,
                            node_out_feats=node_out_feats,
                            edge_in_feats=edge_in_feats,
                            message_hidden_feats=message_hidden_feats,
                            num_message_passing=num_message_passing,
                            shared_message_passing_weights=shared_message_passing_weights)
        
        self.readout = AvgPooling()

        self.predict = nn.Sequential(
            nn.Linear(node_out_feats,node_out_feats),
            nn.ReLU(),
            nn.Dropout(p=drop_out_rate),
            nn.Linear(node_out_feats,node_out_feats),
            nn.ReLU(),
            nn.Dropout(p=drop_out_rate),
            nn.Linear(node_out_feats,num_tasks)
        )

    def forward(self, g, node_feats, edge_feats):
        node_hidden_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_hidden_feats)
        return self.predict(graph_feats)

from utils import _train, _eval, _predict
class SAMPNN:
    def __init__(self,  
                 node_out_feats=64,
                 message_hidden_feats=128,
                 num_message_passing=6,
                 drop_out_rate=0,
                 shared_message_passing_weights=True,
                 cuda=False,
                 metrics='rmse'):
        
        self.cuda = cuda
        self.node_out_feats = node_out_feats
        self.message_hidden_feats = message_hidden_feats
        self.num_message_passing = num_message_passing
        self.drop_out_rate = drop_out_rate
        self.shared_message_passing_weights = shared_message_passing_weights
        self.fitted = False
        self.metrics = metrics

    def fit(self,
            train_loader,
            epochs=50,
            learning_rate=0.001):
        _, ex_g, _, ex_masks = next(iter(train_loader))
        edge_in_feats = ex_g.edge_attr_schemes()['edge_attr'].shape[0]
        node_in_feats = ex_g.node_attr_schemes()['x'].shape[0]
        num_tasks = ex_masks.shape[0]
        self.model = SAMPNPredictor(edge_in_feats,
                                    node_in_feats,
                                    node_out_feats=self.node_out_feats,
                                    message_hidden_feats=self.message_hidden_feats,
                                    num_tasks=num_tasks,
                                    num_message_passing=self.num_message_passing,
                                    drop_out_rate=self.drop_out_rate,
                                    shared_message_passing_weights=self.shared_message_passing_weights)
        _train(self.model, 
                train_loader, 
                learning_rate=learning_rate, 
                cuda=self.cuda, 
                epochs=epochs, 
                metrics=self.metrics, 
                optimizer='adam',
                use_node_feat=True,
                use_edge_feat=True)

        self.fitted = True

    def predict(self,
                test_graphs):
        if not self.fitted:
            print("Model has not been trained yet.")
        else:
            bg = batch(test_graphs)
            return _predict(self.model, bg, self.cuda, 
                            use_node_feat=True, use_edge_feat=True)

    def evaluate(self,
                 val_data_loader):
        if not self.fitted:
            print("Model has not been trained yet.")
        else:
            return _eval(self.model,
                        val_data_loader, 
                        metrics=self.metrics, 
                        cuda=self.cuda,
                        use_node_feat=True, use_edge_feat=True)
