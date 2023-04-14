import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, Dropout
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn.aggr import GraphMultisetTransformer

from MoleculeACE.models.utils import GNN, register_model


@register_model('GIN')
class GIN(GNN):
    def __init__(self, node_in_feats=37, node_hidden=128, edge_in_feats=6, edge_hidden=256, dropout=0.4, aggr='transformer', transformer_heads=8, transformer_hidden=128,
                 lr=0.0005, num_class=1, loss_fn=nn.MSELoss(), save_path=None, *args, **kwargs):
        super().__init__()
        self.model = GINVirtual_node(node_in_feats=node_in_feats, node_hidden=node_hidden, edge_in_feats=edge_in_feats, edge_hidden=edge_hidden,
                                     dropout=dropout, aggr=aggr, transformer_heads=transformer_heads, transformer_hidden=transformer_hidden, num_class=num_class)
        self.critic = GINVirtual_node(node_in_feats=node_in_feats, node_hidden=node_hidden, edge_in_feats=edge_in_feats, edge_hidden=edge_hidden,
                                      dropout=dropout, aggr=aggr, transformer_heads=transformer_heads, transformer_hidden=transformer_hidden, num_class=num_class, critic=True)
        self.name = 'GIN'
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.critic.parameters()), lr=lr, weight_decay=1e-16)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.critic.to(self.device)
        self.save_path = save_path

    def __repr__(self):
        return f"{self.model}"


class GINVirtual_node(torch.nn.Module):
    def __init__(self, node_in_feats=37, node_hidden=128, edge_in_feats=6, edge_hidden=256, dropout=0.4, num_layers=3, aggr='transformer', transformer_heads=8,
                 transformer_hidden=128, batch_norm=False, num_class=1, critic=False):
        super(GINVirtual_node, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.atom_encoder = Sequential(Linear(node_in_feats, node_hidden), ReLU())
        self.virtualnode_embedding = torch.nn.Embedding(1, node_hidden)  # set the initial virtual node embedding to 0.

        self.batch_norm = batch_norm
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GINConv(node_hidden, edge_in_feats, edge_hidden))
            self.batch_norms.append(torch.nn.BatchNorm1d(node_hidden))  # batchnorm can lead to training instability https://stackoverflow.com/questions/33962226/common-causes-of-nans-during-training-of-neural-networks

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(node_hidden, 2 * node_hidden), torch.nn.BatchNorm1d(2 * node_hidden), torch.nn.ReLU(),
                                                                 torch.nn.Linear(2 * node_hidden, node_hidden), torch.nn.BatchNorm1d(node_hidden), torch.nn.ReLU()))

        self.aggr = aggr
        self.num_class = num_class
        if self.aggr == 'transformer':
            self.transformer = GraphMultisetTransformer(in_channels=node_hidden, hidden_channels=transformer_hidden, out_channels=node_hidden, num_heads=transformer_heads)
        else:
            self.pool = global_mean_pool

        self.critic = critic
        if critic:
            self.net_y = torch.nn.Sequential(nn.Linear(node_hidden, node_hidden), nn.ReLU(inplace=True), nn.Linear(node_hidden, node_hidden), nn.ReLU(inplace=True),
                                             nn.Linear(node_hidden, 1), )
            self.fusion = torch.nn.Linear(2, 1)
        else:
            self.lin = Sequential(Linear(node_hidden, node_hidden), Dropout(dropout), Linear(node_hidden, node_hidden), Dropout(dropout), Linear(node_hidden, node_hidden),
                                  Dropout(dropout), Linear(node_hidden, num_class))

    def forward(self, x, edge_index, edge_attr, batch, y_hat=None, loss=None):
        vn_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))  # virtual node embeddings for graphs
        h_list = [self.atom_encoder(x)]

        for layer in range(self.num_layers):
            h_list[layer] = h_list[layer] + vn_embedding[batch]              # add message from virtual nodes to graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)      # Message passing among graph nodes

            if self.batch_norm:
                h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.dropout, training=self.training)       # remove relu for the last layer
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
            h_list.append(h)

            if layer < self.num_layers - 1:  # update the virtual nodes
                vn_embedding_tmp = global_add_pool(h_list[layer], batch) + vn_embedding                                             # add message from graph nodes to virtual nodes
                vn_embedding = F.dropout(self.mlp_virtualnode_list[layer](vn_embedding_tmp), self.dropout, training=self.training)  # transform virtual nodes using MLP

        h_graph = self.transformer(h_list[-1], index=batch, edge_index=edge_index) if self.aggr == 'transformer' else self.pool(h_list[-1], batch)

        if self.critic:
            if self.num_class > 1:
                xy = self.net_y(h_graph.unsqueeze(dim=-2) * y_hat.unsqueeze(dim=-1))
            else:
                xy = self.net_y(h_graph * y_hat)
            loss = loss.unsqueeze(dim=-1)
            output = torch.cat([xy, loss], dim=-1)
            output = self.fusion(output)
            return torch.sigmoid(output)

        return self.lin(h_graph)


class GINConv(MessagePassing):
    def __init__(self, hidden_dim, edge_in_feats, edge_hidden):
        super(GINConv, self).__init__(aggr="add")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(hidden_dim, 2 * hidden_dim), torch.nn.BatchNorm1d(2 * hidden_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * hidden_dim, hidden_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.edge_network = Sequential(Linear(edge_in_feats, edge_hidden), ReLU(), Linear(edge_hidden, hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_network(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
