from typing import Optional
import torch

@torch.no_grad()
def select_feature_combinations_corr(x: torch.Tensor, fs_feature_number: int) -> torch.Tensor:
    x_centered = x - x.mean(dim=0, keepdim=True)
    x_norm = x_centered / torch.clamp(x_centered.norm(dim=0, keepdim=True), min=1e-12)
    score = torch.abs(x_norm.T @ x_norm)
    score.fill_diagonal_(1.0)
    _, top_indices = torch.topk(score, k=fs_feature_number, dim=1, largest=True, sorted=True)
    sorted_indices, _ = torch.sort(top_indices, dim=1)
    return torch.unique(sorted_indices, dim=0).long()

@torch.no_grad()
def _entropy_from_counts(counts: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    prob = counts / torch.clamp(counts.sum(dim=-1, keepdim=True), min=eps)
    return -(prob * torch.log2(torch.clamp(prob, min=eps))).sum(dim=-1)

@torch.no_grad()
def _feature_feature_su_hist(x: torch.Tensor, bins: int=16, verbose: bool=True) -> torch.Tensor:
    n, d = x.shape
    x_bin = torch.clamp((x * bins).long(), min=0, max=bins - 1)
    feature_counts = torch.zeros((d, bins), device=x.device, dtype=torch.float32)
    src = torch.ones((d, n), device=x.device, dtype=torch.float32)
    feature_counts.scatter_add_(1, x_bin.T.contiguous(), src)
    h = _entropy_from_counts(feature_counts)
    su = torch.empty((d, d), device=x.device, dtype=torch.float32)
    for i in range(d):
        joint_code = x_bin[:, i].unsqueeze(1) * bins + x_bin
        joint_counts = torch.zeros((d, bins * bins), device=x.device, dtype=torch.float32)
        joint_counts.scatter_add_(1, joint_code.T.contiguous(), src)
        h_joint = _entropy_from_counts(joint_counts)
        mi = torch.clamp(h[i] + h - h_joint, min=0.0)
        su[i] = 2.0 * mi / torch.clamp(h[i] + h, min=1e-12)
        if verbose and ((i + 1) % 50 == 0 or i + 1 == d):
            pass
    su.fill_diagonal_(1.0)
    return su

@torch.no_grad()
def _feature_label_su_hist(x: torch.Tensor, y: torch.Tensor, bins: int=16, verbose: bool=True) -> torch.Tensor:
    if y is None:
        raise ValueError('y')
    n, d = x.shape
    _, l = y.shape
    x_bin = torch.clamp((x * bins).long(), min=0, max=bins - 1)
    y_bin = (y > 0.5).float()
    feature_counts = torch.zeros((d, bins), device=x.device, dtype=torch.float32)
    src = torch.ones((d, n), device=x.device, dtype=torch.float32)
    feature_counts.scatter_add_(1, x_bin.T.contiguous(), src)
    h_feature = _entropy_from_counts(feature_counts)
    label_counts = torch.stack([y_bin.sum(dim=0), n - y_bin.sum(dim=0)], dim=1)
    h_label = _entropy_from_counts(label_counts)
    valid_label = h_label > 1e-12
    if not valid_label.any():
        return torch.zeros(d, device=x.device, dtype=torch.float32)
    h_joint = torch.zeros((d, l), device=x.device, dtype=torch.float32)
    n_float = float(n)
    for b in range(bins):
        mask = (x_bin == b).float()
        pos_counts = mask.T @ y_bin
        total_counts = feature_counts[:, b].unsqueeze(1)
        neg_counts = total_counts - pos_counts
        for counts in (pos_counts, neg_counts):
            prob = counts / n_float
            h_joint = h_joint - prob * torch.log2(torch.clamp(prob, min=1e-12))
    if verbose:
        pass
    mi = torch.clamp(h_feature.unsqueeze(1) + h_label.unsqueeze(0) - h_joint, min=0.0)
    su_fl = 2.0 * mi / torch.clamp(h_feature.unsqueeze(1) + h_label.unsqueeze(0), min=1e-12)
    return su_fl[:, valid_label].mean(dim=1)

@torch.no_grad()
def select_feature_combinations_su_hist(x: torch.Tensor, fs_feature_number: int, bins: int=16, verbose: bool=True) -> torch.Tensor:
    su = _feature_feature_su_hist(x, bins=bins, verbose=verbose)
    _, top_indices = torch.topk(su, k=fs_feature_number, dim=1, largest=True, sorted=True)
    sorted_indices, _ = torch.sort(top_indices, dim=1)
    return torch.unique(sorted_indices, dim=0).long()

@torch.no_grad()
def select_feature_combinations_su_label_hist(x: torch.Tensor, y: torch.Tensor, fs_feature_number: int, bins: int=16, label_weight: float=0.1, verbose: bool=True) -> torch.Tensor:
    if y is None:
        raise ValueError('y')
    label_weight = float(label_weight)
    if not 0.0 <= label_weight <= 1.0:
        raise ValueError('label_weight')
    su_ff = _feature_feature_su_hist(x, bins=bins, verbose=verbose)
    rel_fl = _feature_label_su_hist(x, y, bins=bins, verbose=verbose)
    rel_fl = rel_fl / torch.clamp(rel_fl.max(), min=1e-12)
    rel_pair = 0.5 * (rel_fl.view(-1, 1) + rel_fl.view(1, -1))
    score = (1.0 - label_weight) * su_ff + label_weight * rel_pair
    score.fill_diagonal_(1.0)
    _, top_indices = torch.topk(score, k=fs_feature_number, dim=1, largest=True, sorted=True)
    sorted_indices, _ = torch.sort(top_indices, dim=1)
    return torch.unique(sorted_indices, dim=0).long()

@torch.no_grad()
def build_feature_combinations(x: torch.Tensor, fs_feature_number: int, y: Optional[torch.Tensor]=None, method: str='su_label_hist', bins: int=16, label_weight: float=0.1, verbose: bool=True) -> torch.Tensor:
    if fs_feature_number < 1:
        raise ValueError('fs_feature_number')
    if fs_feature_number > x.shape[1]:
        raise ValueError('fs_feature_number')
    if method == 'corr':
        return select_feature_combinations_corr(x, fs_feature_number)
    if method == 'su_hist':
        return select_feature_combinations_su_hist(x, fs_feature_number, bins=bins, verbose=verbose)
    if method == 'su_label_hist':
        return select_feature_combinations_su_label_hist(x, y, fs_feature_number, bins=bins, label_weight=label_weight, verbose=verbose)
    raise ValueError(method)
