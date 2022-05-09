import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from graphtransformer import TransformerConv
from loss_function import Mish


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, bond_features_dim, aggr="add"):
        super(GINConv, self).__init__(aggr=aggr, node_dim=0)
        # multi-layer perceptron
        self.bond_features_dim = bond_features_dim
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding = torch.nn.Linear(bond_features_dim, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), self.bond_features_dim)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0).to(torch.float32)
        edge_embeddings = self.edge_embedding(edge_attr)

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, bond_features_dim, aggr="add"):
        super(GCNConv, self).__init__(aggr=aggr, node_dim=0)

        self.emb_dim = emb_dim
        self.bond_features_dim = bond_features_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding = torch.nn.Linear(bond_features_dim, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), self.bond_features_dim)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0).to(torch.float32)

        edge_embeddings = self.edge_embedding(edge_attr)

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, bond_features_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__(aggr=aggr, node_dim=0)

        self.emb_dim = emb_dim
        self.bond_features_dim = bond_features_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))
        self.edge_embedding = torch.nn.Linear(bond_features_dim, heads * emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.mish = Mish()
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), self.bond_features_dim)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0).to(torch.float32)
        edge_embeddings = self.edge_embedding(edge_attr)
        x = self.weight_linear(x)
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_j += edge_attr
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        # alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = self.mish(alpha)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, bond_features_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__(aggr=aggr, node_dim=0)
        self.emb_dim = emb_dim
        self.bond_features_dim = bond_features_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding = torch.nn.Linear(bond_features_dim, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), self.bond_features_dim)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0).to(torch.float32)
        edge_embeddings = self.edge_embedding(edge_attr)
        x = self.linear(x)

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class MATConv(MessagePassing):
    def __init__(self, emb_dim, bond_features_dim, heads):
        super(MATConv, self).__init__()


class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, atom_feat_dim, bond_feat_dim, JK="last", drop_ratio=0, gnn_type="gin",
                 pe=False, pe_dim=8):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.atom_feat_dim = atom_feat_dim
        self.bond_feat_dim = bond_feat_dim
        self.JK = JK
        self.mish = Mish()
        self.wpe = pe
        self.pe_feat_dim = pe_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding = torch.nn.Linear(atom_feat_dim, emb_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
        # self.pe_embedding = torch.nn.Linear(self.pe_feat_dim, emb_dim, bias=False)
        # torch.nn.init.xavier_uniform_(self.pe_embedding.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, bond_feat_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, bond_feat_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim, bond_feat_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim, bond_feat_dim))
            elif gnn_type == 'transformer':
                self.gnns.append(
                    TransformerConv(in_channels=emb_dim, out_channels=int(emb_dim / 8), heads=8, concat=True,
                                    dropout=0.1, edge_dim=bond_feat_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        # self.layer_norms = torch.nn.ModuleList
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            # self.layer_norms.append(torch.nn.LayerNorm(emb_dim))

    def forward(self, x, edge_index, edge_attr, batch=None, pe=None):
        x = self.x_embedding(x)
        # if self.wpe:
        #     pe = self.pe_embedding(pe)
        #     assert x.size() == pe.size()
        #     x = x + pe
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](x=h_list[layer], edge_index=edge_index, edge_attr=edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(self.mish(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("Unknown JK method.")

        return node_representation
