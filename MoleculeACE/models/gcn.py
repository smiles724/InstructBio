"""
Basic graph convolutional network [1] using a transformer as global aggr [2].

1. Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks
2. Baek et al. (2021). Accurate Learning of Graph Representations with Graph Multiset Pooling
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import GraphMultisetTransformer
from MoleculeACE.models.utils import GNN, register_model


@register_model('GCN')
class GCN(GNN):
    def __init__(self, node_feat_in=37, node_hidden=128, transformer_hidden=128, n_conv_layers=4, n_fc_layers=4, fc_hidden=128, lr=0.0005, dropout=0.2,
                 num_class=1, loss_fn=torch.nn.MSELoss(), save_path=None, *args, **kwargs):
        super().__init__()
        self.model = GCNmodel(node_feat_in=node_feat_in, node_hidden=node_hidden, n_conv_layers=n_conv_layers, transformer_hidden=transformer_hidden, fc_hidden=fc_hidden,
                              n_fc_layers=n_fc_layers, dropout=dropout, num_class=num_class)
        self.critic = GCNmodel(node_feat_in=node_feat_in, node_hidden=node_hidden, n_conv_layers=n_conv_layers, transformer_hidden=transformer_hidden, fc_hidden=fc_hidden,
                               n_fc_layers=n_fc_layers, dropout=dropout, num_class=num_class, critic=True)
        self.name = 'GCN'
        self.loss_fn = loss_fn
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.critic.parameters()), lr=lr, weight_decay=1e-16)
        self.model.to(self.device)
        self.critic.to(self.device)
        self.save_path = save_path

    def __repr__(self):
        return f"{self.model}"


class GCNmodel(torch.nn.Module):
    def __init__(self, node_feat_in=37, n_conv_layers=4, n_fc_layers=1, node_hidden=128, fc_hidden=128, transformer_hidden=128, dropout=0.2, num_class=1, critic=False,
                 *args, **kwargs):
        super().__init__()
        self.num_class = num_class

        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(node_feat_in, node_hidden))
        for k in range(n_conv_layers - 1):
            self.conv_layers.append(GCNConv(node_hidden, node_hidden))
        self.transformer = GraphMultisetTransformer(in_channels=node_hidden, hidden_channels=transformer_hidden, out_channels=fc_hidden, num_heads=8)
        self.fc = torch.nn.ModuleList()
        for k in range(n_fc_layers):
            self.fc.append(Linear(fc_hidden, fc_hidden))
        self.dropout = Dropout(dropout)
        self.critic = critic
        if critic:
            self.net_y = torch.nn.Sequential(Linear(fc_hidden, fc_hidden), ReLU(inplace=True), Linear(fc_hidden, fc_hidden), ReLU(inplace=True),
                                             Linear(fc_hidden, 1), )
            self.fusion = torch.nn.Linear(2, 1)
        else:
            self.out = Linear(fc_hidden, num_class)

    def forward(self, x, edge_index, edge_attr, batch, y_hat=None, loss=None):
        # Conv layers
        h = F.relu(self.conv_layers[0](x.float(), edge_index))
        for k in range(len(self.conv_layers) - 1):
            h = F.relu(self.conv_layers[k + 1](h, edge_index))

        # Global graph aggr with a transformer
        h = self.transformer(h, index=batch, edge_index=edge_index)

        # Apply a fully connected layer.
        for k in range(len(self.fc)):
            h = F.relu(self.fc[k](h))
            h = self.dropout(h)

        # Apply a final (linear) classifier.
        if self.critic:
            if self.num_class > 1:
                xy = self.net_y(h.unsqueeze(dim=-2) * y_hat.unsqueeze(dim=-1))
            else:
                xy = self.net_y(h * y_hat)
            loss = loss.unsqueeze(dim=-1)
            output = torch.cat([xy, loss], dim=-1)
            output = self.fusion(output)
            return torch.sigmoid(output)

        return self.out(h)
