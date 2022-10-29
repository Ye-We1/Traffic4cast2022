import os
import sys

import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath("../"))

import statistics
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
import tqdm
from IPython.core.display import HTML
from IPython.display import display
from torch import nn
from torch_geometric.nn import MessagePassing
from pathlib import Path
import numpy as np

import t4c22
from t4c22.metric.masked_crossentropy import get_weights_from_class_fractions
from t4c22.misc.t4c22_logging import t4c_apply_basic_logging_config
from t4c22.t4c22_config import class_fractions
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset

import math

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Parameter
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn import inits
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import MLP
from torch_geometric.typing import Adj, OptTensor



class SparseLinear(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        inits.kaiming_uniform(self.weight, fan=self.in_channels,
                              a=math.sqrt(5))
        inits.uniform(self.in_channels, self.bias)

    def forward(self, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        # propagate_type: (weight: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, weight=self.weight,
                             edge_weight=edge_weight, size=None)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, weight_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            return weight_j
        else:
            return edge_weight.view(-1, 1) * weight_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              weight: Tensor) -> Tensor:
        return matmul(adj_t, weight, reduce=self.aggr)

class LINKX(torch.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, num_edge_layers: int = 1,
                 num_node_layers: int = 1, dropout: float = 0.):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_layers = num_edge_layers

        self.edge_lin = SparseLinear(num_nodes, hidden_channels)
        if self.num_edge_layers > 1:
            self.edge_norm = BatchNorm1d(hidden_channels)
            channels = [hidden_channels] * num_edge_layers
            self.edge_mlp = MLP(channels, dropout=0., act_first=True)

        channels = [in_channels] + [hidden_channels] * num_node_layers
        self.node_mlp = MLP(channels, dropout=0., act_first=True)

        self.cat_lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

        channels = [hidden_channels] * num_layers + [out_channels]
        self.final_mlp = MLP(channels, dropout=dropout, act_first=True)

        self.reset_parameters()

    def reset_parameters(self):
            self.edge_lin.reset_parameters()
            if self.num_edge_layers > 1:
                self.edge_norm.reset_parameters()
                self.edge_mlp.reset_parameters()
            self.node_mlp.reset_parameters()
            self.cat_lin1.reset_parameters()
            self.cat_lin2.reset_parameters()
            self.final_mlp.reset_parameters()


    def forward(self, x: OptTensor, edge_index: Adj,
                    edge_weight: OptTensor = None) -> Tensor:
            
            out = self.edge_lin(edge_index, edge_weight)
            if self.num_edge_layers > 1:
                out = out.relu_()
                out = self.edge_norm(out)
                out = self.edge_mlp(out)

            out = out + self.cat_lin1(out)

            if x is not None:

                x = self.node_mlp(x)
                out = out + x
                out = out + self.cat_lin2(x)

            return self.final_mlp(out.relu_())


    def __repr__(self) -> str:
            return (f'{self.__class__.__name__}(num_nodes={self.num_nodes}, '
                    f'in_channels={self.in_channels}, '
                    f'out_channels={self.out_channels})')


class Swish(nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.swish = Swish()

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = self.swish(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return x

def train(model, predictor, dataset, optimizer, batch_size, device):
    model.train()

    losses = []
    optimizer.zero_grad()

    for data in tqdm.tqdm(
        torch_geometric.loader.dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        "train",
        total=len(dataset) // batch_size,
    ):

        data = data.to(device)

        data.x = data.x.nan_to_num(-1)

        h = model(data.x, data.edge_index)
        assert (h.isnan()).sum() == 0, h
        x_i = torch.index_select(h, 0, data.edge_index[0])
        x_j = torch.index_select(h, 0, data.edge_index[1])

        y_hat = predictor(x_i, x_j)

        y = data.y.nan_to_num(-1)
        y = y.long()

        loss_f = torch.nn.CrossEntropyLoss(weight=city_class_weights, ignore_index=-1)
        loss = loss_f(y_hat, y)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.cpu().item())

    return losses


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath("../"))
    t4c_apply_basic_logging_config(loglevel="DEBUG")
    BASEDIR = Path("./data")

    cities = ["melbourne", "london", "madrid"]
    for city in cities:
        dataset = T4c22GeometricDataset(root=BASEDIR, city=city, split="train", cachedir=Path("/tmp/processed"))
        train_dataset = dataset

        city_class_fractions = class_fractions[city]
        city_class_weights = torch.tensor(get_weights_from_class_fractions([city_class_fractions[c] for c in ["green", "yellow", "red"]])).float()

        hidden_channels = 256
        num_layers = 3
        batch_size = 1
        eval_steps = 1
        epochs = 150
        runs = 1
        dropout = 0.4
        num_edge_classes = 3
        num_features = 4

        device = 0
        device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)


        city_class_weights = city_class_weights.to(device)

        if city == "melbourne":
            model = LINKX(49510, num_features, hidden_channels, hidden_channels, num_layers, num_layers, num_layers, dropout)
            model = model.to(device)
        if city == "madrid":
            model = LINKX(63397, num_features, hidden_channels, hidden_channels, num_layers, num_layers, num_layers, dropout)
            model = model.to(device)
        if city == "london":
            model = LINKX(59110, num_features, hidden_channels, hidden_channels, num_layers, num_layers, num_layers, dropout)
            model = model.to(device)

        predictor = LinkPredictor(hidden_channels, hidden_channels, num_edge_classes, num_layers, dropout).to(device)

        train_losses = defaultdict(lambda: [])
        val_losses = defaultdict(lambda: -1)

        bar = 2.0

        for run in tqdm.tqdm(range(runs), desc="runs", total=runs):
            predictor.reset_parameters()
            optimizer = torch.optim.AdamW(
                    [
                        {"params": model.parameters()},
                        {"params": predictor.parameters()}
                    ],
                    lr=3e-4,
                    weight_decay=0.001
                )

            for epoch in tqdm.tqdm(range(1, 1 + epochs), "epochs", total=epochs):
                torch.cuda.empty_cache()
                losses = train(model, predictor, dataset=train_dataset, optimizer=optimizer, batch_size=batch_size, device=device)
                train_losses[(run, epoch)] = losses

                print(statistics.mean(losses))

                torch.save(model.state_dict(), f"GNN_model_{city}_{epoch:03d}.pt")
                torch.save(predictor.state_dict(), f"GNN_predictor_{city}_{epoch:03d}.pt")
