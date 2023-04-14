"""
Message Passing Neural Network for molecular property prediction using continuous kernel-based convolution
(edge-conditioned convolution) [1] and global graph aggr using a graph multiset transformer [2] instead of
the Set2Set method used in [1]. I added dropout to make a more robust

[1] Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry
[2] Baek et al. (2021). Accurate Learning of Graph Representations with Graph Multiset Pooling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential, Dropout
from torch_geometric.nn import NNConv
from torch_geometric.nn.aggr import GraphMultisetTransformer
from MoleculeACE.models.utils import GNN, register_model


@register_model('MPNN')
class MPNN(GNN):
    def __init__(self, node_in_feats=37, node_hidden=64, edge_in_feats=6, edge_hidden=128, message_steps=3, dropout=0.2, transformer_heads=8, transformer_hidden=128, fc_hidden=64,
                 n_fc_layers=1, lr=0.0005, num_class=1, loss_fn=nn.MSELoss(), save_path=None, *args, **kwargs):
        super().__init__()
        self.model = MPNNmodel(node_in_feats=node_in_feats, node_hidden=node_hidden, edge_in_feats=edge_in_feats, edge_hidden=edge_hidden, message_steps=message_steps,
                               dropout=dropout, transformer_heads=transformer_heads, transformer_hidden=transformer_hidden, fc_hidden=fc_hidden, n_fc_layers=n_fc_layers,
                               num_class=num_class)
        self.critic = MPNNmodel(node_in_feats=node_in_feats, node_hidden=node_hidden, edge_in_feats=edge_in_feats, edge_hidden=edge_hidden, message_steps=message_steps,
                                dropout=dropout, transformer_heads=transformer_heads, transformer_hidden=transformer_hidden, fc_hidden=fc_hidden, n_fc_layers=n_fc_layers,
                                num_class=num_class, critic=True)
        self.name = 'MPNN'
        self.loss_fn = loss_fn
        self.save_path = save_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.critic.parameters()), lr=lr, weight_decay=1e-16)
        self.critic.to(self.device)
        self.model.to(self.device)

    def __repr__(self):
        return f"{self.model}"


class MPNNmodel(nn.Module):
    def __init__(self, node_in_feats=37, node_hidden=64, edge_in_feats=6, edge_hidden=128, message_steps=3, dropout=0.2, transformer_heads=8, transformer_hidden=128, fc_hidden=64,
                 n_fc_layers=1, num_class=1, critic=False, *args, **kwargs):
        super().__init__()
        self.node_hidden = node_hidden
        self.messsage_steps = message_steps
        self.node_in_feats = node_in_feats
        self.project_node_feats = Sequential(Linear(node_in_feats, node_hidden), ReLU())
        edge_network = Sequential(Linear(edge_in_feats, edge_hidden), ReLU(), Linear(edge_hidden, node_hidden * node_hidden))
        self.gnn_layer = NNConv(in_channels=node_hidden, out_channels=node_hidden, nn=edge_network, aggr='add')

        # The GRU as used in [1]
        self.gru = GRU(node_hidden, node_hidden)

        # Global aggr using a transformer, https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.GraphMultisetTransformer.html
        # https://pytorch-geometric.readthedocs.io/en/2.2.0/modules/nn.html?highlight=GraphMultisetTransformer#torch_geometric.nn.aggr.GraphMultisetTransformer
        self.transformer = GraphMultisetTransformer(in_channels=node_hidden, hidden_channels=transformer_hidden, out_channels=fc_hidden, num_heads=transformer_heads)

        self.fc = nn.ModuleList()
        for k in range(n_fc_layers):
            self.fc.append(Linear(fc_hidden, fc_hidden))

        self.dropout = Dropout(dropout)
        self.critic = critic
        if critic:
            self.net_y = torch.nn.Sequential(Linear(fc_hidden, fc_hidden), ReLU(inplace=True), Linear(fc_hidden, fc_hidden), ReLU(inplace=True), Linear(fc_hidden, 1), )
            self.fusion = torch.nn.Linear(2, 1)
        else:
            self.lin2 = torch.nn.Linear(fc_hidden, num_class)

    def forward(self, x, edge_index, edge_attr, batch, y_hat=None, loss=None):
        node_feats = self.project_node_feats(x)
        hidden_feats = node_feats.unsqueeze(0)

        # Perform n message passing steps using edge-conditioned convolution as pass it through a GRU
        for _ in range(self.messsage_steps):
            node_feats = F.relu(self.gnn_layer(node_feats, edge_index, edge_attr))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)

        # perform global aggr using a multiset transformer to get graph-wise hidden embeddings
        out = self.transformer(node_feats, index=batch, edge_index=edge_index)  # we use torch_geometric 2.2.0 instead of 2.0.4

        for k in range(len(self.fc)):
            out = F.relu(self.fc[k](out))
            out = self.dropout(out)

        # Apply a final (linear) classifier.
        if self.critic:
            if self.num_class > 1:
                xy = self.net_y(out.unsqueeze(dim=-2) * y_hat.unsqueeze(dim=-1))
            else:
                xy = self.net_y(out * y_hat)
            loss = loss.unsqueeze(dim=-1)
            output = torch.cat([xy, loss], dim=-1)
            output = self.fusion(output)
            return torch.sigmoid(output)

        return self.lin2(out)
