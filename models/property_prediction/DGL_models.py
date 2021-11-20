from dgllife.model.model_zoo.gat_predictor import GATPredictor
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor
from dgllife.model.model_zoo.mgcn_predictor import MGCNPredictor
from dgllife.model.model_zoo.mpnn_predictor import MPNNPredictor
from dgllife.model.model_zoo.schnet_predictor import SchNetPredictor
from dgllife.model.model_zoo.weave_predictor import WeavePredictor

from dgl import batch

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
                optimizer='adam')

    def predict(self,
                test_graphs):
        if not self.fitted:
            print("Model has not been trained yet.")
        else:
            bg = batch(test_graphs)
            return _predict(self.model, bg, self.cuda)

    def evaluate(self,
                 val_data_loader):
        if not self.fitted:
            print("Model has not been trained yet.")
        else:
            return _eval(self.model,
                        val_data_loader, 
                        metrics=self.metrics, 
                        cuda=self.cuda)