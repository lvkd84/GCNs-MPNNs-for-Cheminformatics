import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

import numpy as np

from dgllife.utils import Meter
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgl import batch

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
    """
    D-MPNN model.
    Parameters
    ----------
    in_edge_feats
        Number of input edge features
    in_node_feats
        Number of input node features
    out_node_feats
        Number of output node features
    hidden_edge_feats
        Number of edge hidden features
    num_tasks
        Number of prediction tasks
    num_steps_passing
        Number of message passing layers
    drop_out_rate
        The drop-out rate at the fully-connected layers
    """  
    def __init__(self,
                 in_edge_feats,
                 in_node_feats,  
                 out_node_feats=64,
                 hidden_edge_feats=128,
                 num_tasks=1,
                 num_steps_passing=6,
                 drop_out_rate=0):
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
            nn.Dropout(p=drop_out_rate),
            nn.Linear(out_node_feats,num_tasks)
        )
    
    def forward(self, g, node_feats, edge_feats):
        node_feats = self.gnn(g, node_feats, edge_feats)
        g_feats = self.readout(g, node_feats)
        return self.predict(g_feats)


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RMSELoss, self).__init__()
        self.reduction = reduction 

    def forward(self,input,target):
        return torch.sqrt(F.mse_loss(input, target, reduction=self.reduction))

metrics_dic = {
    'rmse': RMSELoss
}

class DMPNN:
    """
    D-MPNN model.
    Parameters
    ----------
    out_node_feats
        Number of output node features
    hidden_edge_feats
        Number of edge hidden features
    num_steps_passing
        Number of message passing layers
    drop_out_rate
        The drop-out rate at the fully-connected layers
    """  
    def __init__(self,  
                 out_node_feats=64,
                 hidden_edge_feats=128,
                 num_steps_passing=6,
                 drop_out_rate=0,
                 cuda=False,
                 metrics='rmse'):
        
        self.cuda = cuda
        self.out_node_feats = out_node_feats
        self.hidden_edge_feats = hidden_edge_feats
        self.num_steps_passing = num_steps_passing
        self.drop_out_rate = drop_out_rate
        self.fitted = False
        self.metrics = metrics
        self.loss = metrics_dic[metrics]()

    def __predict__(self, model, bg, device):
        bg = bg.to(device)
        node_feats = bg.ndata.pop('x').to(device)
        edge_feats = bg.edata.pop('edge_attr').to(device)
        return model(bg, node_feats, edge_feats)

    def fit(self,
            train_loader,
            epochs=50,
            learning_rate=0.001):
        _, ex_g, _, ex_masks = next(iter(train_loader))
        in_edge_feats = ex_g.edge_attr_schemes()['edge_attr'].shape[0]
        in_node_feats = ex_g.node_attr_schemes()['x'].shape[0]
        num_tasks = ex_masks.shape[0]
        self.model = DMPNNPredictor(in_edge_feats,
                                    in_node_feats,
                                    self.out_node_feats,
                                    self.hidden_edge_feats,
                                    num_tasks,
                                    self.num_steps_passing,
                                    self.drop_out_rate)
        if self.cuda:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                print("No cuda found. Train on CPU instead")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')

        self.model.to(device)

        loss_criterion = self.loss
        optimizer = torch.optim.Adam(self.model.parameters(),lr=learning_rate)

        for epoch in range(epochs):
            print("Epoch:", epoch)
            self.model.train()
            train_meter = Meter()
            for batch_id, batch_data in enumerate(train_loader):
                _, bg, labels, masks = batch_data
                labels = labels.to(device)
                masks = masks.to(device)
                prediction = self.__predict__(self.model, bg, device)
                loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_meter.update(prediction, labels, masks)
                if batch_id % 100 == 0:
                    print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                        epoch + 1, 10, batch_id + 1, len(train_loader), loss.item()))
            train_score = np.mean(train_meter.compute_metric(self.metrics))
            print('epoch {:d}/{:d}, training {} {:.4f}'.format(
                epoch + 1, epochs, 'score', train_score))
        
        print("Finished training.")
        self.fitted = True

    def predict(self,
                test_graphs):
        if not self.fitted:
            print("Model has not been trained yet.")
        else:
            if self.cuda:
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    print("No cuda found. Train on CPU instead")
                    device = torch.device('cpu')
            else:
                device = torch.device('cpu')
            bg = batch(test_graphs)
            self.model.eval()
            return self.__predict__(self.model, bg, device)

    def evaluate(self,
                 val_data_loader):
        if not self.fitted:
            print("Model has not been trained yet.")
        else:
            if self.cuda:
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    print("No cuda found. Train on CPU instead")
                    device = torch.device('cpu')
            else:
                device = torch.device('cpu')
            eval_meter = Meter()
            for _, batch_data in enumerate(val_data_loader):
                _, bg, labels, masks = batch_data
                labels = labels.to(device)
                masks = masks.to(device)
                prediction = self.__predict__(self.model, bg, device)
                eval_meter.update(prediction, labels, masks)
            eval_score = np.mean(eval_meter.compute_metric(self.metrics))
            return eval_score