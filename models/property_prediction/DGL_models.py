from dgllife.model.model_zoo.gat_predictor import GATPredictor
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor
from dgllife.model.model_zoo.mgcn_predictor import MGCNPredictor
from dgllife.model.model_zoo.mpnn_predictor import MPNNPredictor
from dgllife.model.model_zoo.schnet_predictor import SchNetPredictor
from dgllife.model.model_zoo.weave_predictor import WeavePredictor

from dgl import batch

from torch.nn import Tanh
from torch.nn.functional import relu

from utils import _train, _eval, _predict

class GAT:

    def __init__(self, hidden_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None,
                 biases=None, predictor_hidden_feats=128, predictor_dropout=0.,
                 cuda=False, metrics='rmse'):
        self.hidden_feats = hidden_feats
        self.num_heads = num_heads
        self.feat_drops = feat_drops
        self.attn_drops = attn_drops
        self.alphas = alphas
        self.residuals = residuals
        self.agg_modes = agg_modes
        self.activations = activations
        self.biases = biases
        self.predictor_hidden_feats = predictor_hidden_feats
        self.predictor_dropout = predictor_dropout
        self.cuda = cuda
        self.metrics = metrics

    def fit(self,
            train_loader,
            epochs=50,
            learning_rate=0.001):
        _, ex_g, _, ex_masks = next(iter(train_loader))
        while not (len(ex_g.node_attr_schemes()) > 0 and len(ex_g.node_attr_schemes()) > 0):
            _, ex_g, _, ex_masks = next(iter(train_loader))
        in_feats = ex_g.node_attr_schemes()['x'].shape[0]
        n_tasks = ex_masks.shape[0]
        self.model = GATPredictor(in_feats=in_feats, 
                                  hidden_feats=self.hidden_feats, 
                                  num_heads=self.num_heads, 
                                  feat_drops=self.feat_drops, 
                                  attn_drops=self.attn_drops, 
                                  alphas=self.alphas, 
                                  residuals=self.residuals, 
                                  agg_modes=self.agg_modes,
                                  activations=self.activations, 
                                  biases=self.biases,
                                  n_tasks=n_tasks,
                                  predictor_hidden_feats=self.predictor_hidden_feats, 
                                  predictor_dropout=self.predictor_dropout)
        _train(self.model, 
                train_loader, 
                learning_rate=learning_rate, 
                cuda=self.cuda, 
                epochs=epochs, 
                metrics=self.metrics, 
                optimizer='adam',
                use_node_feat=True,
                use_edge_feat=False)

    def predict(self,
                test_graphs):
        if not self.fitted:
            print("Model has not been trained yet.")
        else:
            bg = batch(test_graphs)
            return _predict(self.model, bg, self.cuda,
                            use_node_feat=True, use_edge_feat=False)

    def evaluate(self,
                 val_data_loader):
        if not self.fitted:
            print("Model has not been trained yet.")
        else:
            return _eval(self.model,
                        val_data_loader, 
                        metrics=self.metrics, 
                        cuda=self.cuda,
                        use_node_feat=True, use_edge_feat=False)


class GCN:

    def __init__(self, hidden_feats=None, gnn_norm=None, activation=None,
                 residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128,
                 predictor_hidden_feats=128, predictor_dropout=0.,
                 cuda=False, metrics='rmse'):
        self.hidden_feats = hidden_feats
        self.gnn_norm = gnn_norm
        self.activation = activation
        self.residual = residual
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.classifier_hidden_feats = classifier_hidden_feats
        self.predictor_hidden_feats = predictor_hidden_feats
        self.predictor_dropout = predictor_dropout
        self.cuda = cuda
        self.metrics = metrics

    def fit(self,
            train_loader,
            epochs=50,
            learning_rate=0.001):
        _, ex_g, _, ex_masks = next(iter(train_loader))
        while not (len(ex_g.node_attr_schemes()) > 0 and len(ex_g.node_attr_schemes()) > 0):
            _, ex_g, _, ex_masks = next(iter(train_loader))
        in_feats = ex_g.node_attr_schemes()['x'].shape[0]
        n_tasks = ex_masks.shape[0]
        self.model = GCNPredictor(in_feats=in_feats, 
                                  hidden_feats=self.hidden_feats, 
                                  gnn_norm=self.gnn_norm, 
                                  activation=self.activation, 
                                  residual=self.residual, 
                                  batchnorm=self.batchnorm, 
                                  dropout=self.dropout,
                                  n_tasks=n_tasks,
                                  predictor_hidden_feats=self.predictor_hidden_feats, 
                                  predictor_dropout=self.predictor_dropout)
        _train(self.model, 
                train_loader, 
                learning_rate=learning_rate, 
                cuda=self.cuda, 
                epochs=epochs, 
                metrics=self.metrics, 
                optimizer='adam',
                use_node_feat=True,
                use_edge_feat=False)

    def predict(self,
                test_graphs):
        if not self.fitted:
            print("Model has not been trained yet.")
        else:
            bg = batch(test_graphs)
            return _predict(self.model, bg, self.cuda,
                            use_node_feat=True, use_edge_feat=False)

    def evaluate(self,
                 val_data_loader):
        if not self.fitted:
            print("Model has not been trained yet.")
        else:
            return _eval(self.model,
                        val_data_loader, 
                        metrics=self.metrics, 
                        cuda=self.cuda,
                        use_node_feat=True, use_edge_feat=False)


class MPNN:

    def __init__(self,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3,
                 cuda=False, metrics='rmse'):
        self.node_out_feats = node_out_feats
        self.edge_hidden_feats = edge_hidden_feats
        self.num_step_message_passing = num_step_message_passing
        self.num_step_set2set = num_step_set2set
        self.num_layer_set2set = num_layer_set2set
        self.cuda = cuda
        self.metrics = metrics

    def fit(self,
            train_loader,
            epochs=50,
            learning_rate=0.001):
        _, ex_g, _, ex_masks = next(iter(train_loader))
        while not (len(ex_g.node_attr_schemes()) > 0 and len(ex_g.node_attr_schemes()) > 0):
            _, ex_g, _, ex_masks = next(iter(train_loader))
        node_in_feats = ex_g.node_attr_schemes()['x'].shape[0]
        edge_in_feats = ex_g.edge_attr_schemes()['edge_attr'].shape[0]
        n_tasks = ex_masks.shape[0]
        self.model = MPNNPredictor(node_in_feats=node_in_feats, 
                                   edge_in_feats=edge_in_feats, 
                                   node_out_feats=self.node_out_feats, 
                                   edge_hidden_feats=self.edge_hidden_feats, 
                                   n_tasks=n_tasks, 
                                   num_step_message_passing=self.num_step_message_passing, 
                                   num_step_set2set=self.num_step_set2set,
                                   num_layer_set2set=self.num_layer_set2set)
        _train(self.model, 
                train_loader, 
                learning_rate=learning_rate, 
                cuda=self.cuda, 
                epochs=epochs, 
                metrics=self.metrics, 
                optimizer='adam',
                use_node_feat=True,
                use_edge_feat=True)

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

class MGCN:
    pass

class SchNet:
    pass

class WeaveNet:

    def __init__(self,
                 num_gnn_layers=2,
                 gnn_hidden_feats=50,
                 gnn_activation=relu,
                 graph_feats=128,
                 gaussian_expand=True,
                 gaussian_memberships=None,
                 readout_activation=Tanh(),
                 cuda=False, metrics='rmse'):
        self.num_gnn_layers = num_gnn_layers
        self.gnn_hidden_feats = gnn_hidden_feats
        self.gnn_activation = gnn_activation
        self.graph_feats = graph_feats
        self.gaussian_expand = gaussian_expand
        self.gaussian_memberships = gaussian_memberships
        self.readout_activation = readout_activation
        self.cuda = cuda
        self.metrics = metrics

    def fit(self,
            train_loader,
            epochs=50,
            learning_rate=0.001):
        _, ex_g, _, ex_masks = next(iter(train_loader))
        while not (len(ex_g.node_attr_schemes()) > 0 and len(ex_g.node_attr_schemes()) > 0):
            _, ex_g, _, ex_masks = next(iter(train_loader))
        node_in_feats = ex_g.node_attr_schemes()['x'].shape[0]
        edge_in_feats = ex_g.edge_attr_schemes()['edge_attr'].shape[0]
        n_tasks = ex_masks.shape[0]
        self.model = WeavePredictor(node_in_feats=node_in_feats, 
                                    edge_in_feats=edge_in_feats, 
                                    num_gnn_layers=self.num_gnn_layers, 
                                    gnn_hidden_feats=self.gnn_hidden_feats, 
                                    n_tasks=n_tasks, 
                                    gnn_activation=self.gnn_activation, 
                                    graph_feats=self.graph_feats,
                                    gaussian_expand=self.gaussian_expand,
                                    gaussian_memberships=self.gaussian_memberships,
                                    readout_activation=self.readout_activation)
        _train(self.model, 
                train_loader, 
                learning_rate=learning_rate, 
                cuda=self.cuda, 
                epochs=epochs, 
                metrics=self.metrics, 
                optimizer='adam',
                use_node_feat=True,
                use_edge_feat=True)

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
