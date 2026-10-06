import itertools
import numpy as np
import torch
import torch.nn as nn

class NFS(nn.Module):

    def __init__(self, fs_number: int, fs_feature_number: int, number_of_labels: int, number_of_features: int, unique_indices: torch.Tensor, hidden_dim: int=500, defuzzify_eps: float=1e-12):
        super().__init__()
        self.fs_number = fs_number
        self.fs_feature_number = fs_feature_number
        self.number_of_labels = number_of_labels
        self.number_of_features = number_of_features
        self.com_num = int(unique_indices.shape[0])
        self.rule_num = fs_number ** fs_feature_number
        self.defuzzify_eps = float(defuzzify_eps)
        self.register_buffer('unique_indices', unique_indices.long())
        rule_index_np = np.array(list(itertools.product(range(fs_number), repeat=fs_feature_number)), dtype=np.int64)
        self.register_buffer('rule_indices', torch.from_numpy(rule_index_np).long())
        self.central_value_list = nn.Parameter(torch.rand(self.com_num, fs_feature_number, fs_number))
        self.sigma_value_list = nn.Parameter(torch.rand(self.com_num, fs_feature_number, fs_number))
        self.rule_score = nn.Parameter(torch.rand(self.com_num, self.rule_num, number_of_labels))
        self.global_network = nn.Sequential(nn.Linear(number_of_features, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, self.com_num))
        with torch.no_grad():
            self.central_value_list.copy_(torch.sort(self.central_value_list, dim=-1).values)

    def selector(self, x: torch.Tensor) -> torch.Tensor:
        gather_index = self.unique_indices.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.gather(x.unsqueeze(1).expand(-1, self.com_num, -1), dim=2, index=gather_index)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        selected_x = self.selector(inputs)
        centers = torch.sort(self.central_value_list, dim=-1).values
        sigmas = torch.clamp(self.sigma_value_list, min=1e-06)
        membership = torch.exp(-0.5 * ((selected_x.unsqueeze(-1) - centers.unsqueeze(0)) / sigmas.unsqueeze(0)) ** 2)
        batch_size = inputs.shape[0]
        leafs = torch.ones(batch_size, self.com_num, self.rule_num, device=inputs.device, dtype=inputs.dtype)
        for f in range(self.fs_feature_number):
            fuzzy_ids = self.rule_indices[:, f]
            leafs = leafs * membership[:, :, f, :].index_select(dim=2, index=fuzzy_ids)
        gates = self.global_network(inputs)
        leafs_for_defuzz = leafs / torch.clamp(leafs.sum(dim=2, keepdim=True), min=self.defuzzify_eps)
        weighted_leafs = leafs_for_defuzz * gates.unsqueeze(-1)
        logits = torch.einsum('bcr,crl->bl', weighted_leafs, self.rule_score)
        return logits
