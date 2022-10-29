#  Copyright 2022 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import datetime
from functools import partial
from pathlib import Path

import torch

import t4c22
from t4c22.dataloading.t4c22_dataset import T4c22Dataset  # noqa
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset
from t4c22.evaluation.create_submission import create_submission_cc_plain_torch
from t4c22.evaluation.create_submission import create_submission_cc_torch_geometric
from t4c22.evaluation.test_create_submission import apply_model_geometric
# from t4c22.evaluation.test_create_submission import apply_model_plain
# from t4c22.evaluation.test_create_submission import DummyRandomNN_cc
from t4c22.misc.t4c22_logging import t4c_apply_basic_logging_config
from t4c22.t4c22_config import load_basedir

from t4c22.evaluation.test_create_submission import CongestioNN
from t4c22.evaluation.test_create_submission import LinkPredictor
from t4c22.evaluation.test_create_submission import predict_dummy_gnn
from t4c22.evaluation.create_submission import inference_cc_city_torch_geometric_to_pandas
from t4c22.misc.parquet_helpers import write_df_to_parquet
import zipfile
import os

# The submission zip for the core competition must have the following file structure:
#     ```
# london/labels/cc_labels_test.parquet
# madrid/labels/cc_labels_test.parquet
# melbourne/labels/cc_labels_test.parquet
# ```
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

def main(basedir: Path, submission_name: str, model_class, dataset_class, geom=False):
    t4c_apply_basic_logging_config(loglevel="DEBUG")

    cities = ["london", "melbourne", "madrid"]

    config = {}
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    for city in cities:
        test_dataset = dataset_class(root=basedir, city=city, split="test")

        hidden_channels = 256
        num_layers = 3
        dropout = 0.0
        num_edge_classes = 3
        num_features = 4

        device = f"cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

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

        

        if city == "london":

            model.load_state_dict(torch.load(f"checkpoint/GNN_model_london.pt", map_location=device)) 

            predictor.load_state_dict(torch.load(f"checkpoint/GNN_predictor_london.pt", map_location=device))
        
        if city == "melbourne":

            model.load_state_dict(torch.load(f"checkpoint/GNN_model_melbourne.pt", map_location=device)) 

            predictor.load_state_dict(torch.load(f"checkpoint/GNN_predictor_melbourne.pt", map_location=device))

        if city == "madrid":

            model.load_state_dict(torch.load(f"checkpoint/GNN_model_madrid.pt", map_location=device)) 

            predictor.load_state_dict(torch.load(f"checkpoint/GNN_predictor_madrid.pt", map_location=device))

        
        model = model.to(device)
        predictor = predictor.to(device)
        
        config[city] = (test_dataset, partial(apply_model_geometric, device=device, model=model))

        submission = inference_cc_city_torch_geometric_to_pandas(
            predict=partial(predict_dummy_gnn, device=device, model=model, predictor=predictor), test_dataset=test_dataset
        )

        print(submission)

        (basedir / "submission" / submission_name / city / "labels").mkdir(exist_ok=True, parents=True)
        write_df_to_parquet(df=submission, fn=basedir / "submission" / submission_name / city / "labels" / f"cc_labels_test.parquet")

    submission_zip = basedir / "submission" / f"{submission_name}.zip"
    with zipfile.ZipFile(submission_zip, "w") as z:
        for city in cities:
            z.write(
                filename=basedir / "submission" / submission_name / city / "labels" / f"cc_labels_test.parquet",
                arcname=os.path.join(city, "labels", f"cc_labels_test.parquet"),
            )
    print(submission_zip)


if __name__ == "__main__":

    model_class = CongestioNN
    dataset_class = T4c22GeometricDataset
    geom = True

    submission_name = f"{model_class.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    print(submission_name)
    basedir = Path("data")

    main(basedir=basedir, submission_name=submission_name, model_class=model_class, dataset_class=dataset_class, geom=geom)
