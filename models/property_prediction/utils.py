import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from dgllife.utils import Meter

class RMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RMSELoss, self).__init__()
        self.reduction = reduction 

    def forward(self,input,target):
        return torch.sqrt(F.mse_loss(input, target, reduction=self.reduction))

METRICS = {
    'rmse': RMSELoss
}

OPTIMIZERS = {
    'adam': torch.optim.Adam
}

def _predict(model, bg, cuda=False, use_node_feat=True, use_edge_feat=False):
    if cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("No cuda found. Run on CPU instead")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    model.eval()
    bg = bg.to(device)
    if use_node_feat:
        node_feats = bg.ndata.pop('x').to(device)
    if use_edge_feat:
        edge_feats = bg.edata.pop('edge_attr').to(device)
    if use_node_feat and use_edge_feat:
        return model(bg, node_feats, edge_feats)
    elif use_node_feat:
        return model(bg, node_feats)
    else:
        return model(bg, edge_feats)

def _train(model, train_loader, learning_rate=0.001, cuda=False, 
           epochs=50, metrics='rmse', optimizer='adam',
           use_node_feat=True,use_edge_feat=False):
    if cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("No cuda found. Run on CPU instead")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    model.to(device)

    loss_criterion = METRICS[metrics]()
    optimizer = OPTIMIZERS[optimizer](model.parameters(),lr=learning_rate)

    for epoch in range(epochs):
        print("Epoch:", epoch)
        model.train()
        train_meter = Meter()
        for batch_id, batch_data in enumerate(train_loader):
            _, bg, labels, masks = batch_data
            labels = labels.to(device)
            masks = masks.to(device)
            prediction = _predict(model, bg, cuda, use_node_feat, use_edge_feat)
            loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter.update(prediction, labels, masks)
            if batch_id % 100 == 0:
                print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                    epoch + 1, 10, batch_id + 1, len(train_loader), loss.item()))
        train_score = np.mean(train_meter.compute_metric(metrics))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, epochs, 'score', train_score))
    
    print("Finished training.")

def _eval(model, val_data_loader, metrics='rmse', cuda=False,
          use_node_feat=True,use_edge_feat=False):
    if cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("No cuda found. Run on CPU instead")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    eval_meter = Meter()
    for _, batch_data in enumerate(val_data_loader):
        _, bg, labels, masks = batch_data
        labels = labels.to(device)
        masks = masks.to(device)
        prediction = _predict(model, bg, cuda, use_node_feat, use_edge_feat)
        eval_meter.update(prediction, labels, masks)
    eval_score = np.mean(eval_meter.compute_metric(metrics))
    return eval_score