import torch

@torch.no_grad()
def _ascending_ranks(scores: torch.Tensor) -> torch.Tensor:
    n, m = scores.shape
    order = torch.argsort(scores, dim=1, descending=False)
    ranks = torch.empty((n, m), device=scores.device, dtype=torch.float32)
    rank_values = torch.arange(1, m + 1, device=scores.device, dtype=torch.float32).view(1, -1).expand(n, -1)
    ranks.scatter_(1, order, rank_values)
    return ranks

@torch.no_grad()
def do_metric_torch(y_score: torch.Tensor, label: torch.Tensor, threshold: float=0.5) -> torch.Tensor:
    label = (label > 0.5).float()
    n, m = label.shape
    pred = (y_score >= threshold).float()
    hamming_loss = 1.0 - (pred == label).float().mean()
    top_idx = torch.argmax(y_score, dim=1, keepdim=True)
    one_error = 1.0 - label.gather(1, top_idx).mean()
    ranks_asc = _ascending_ranks(y_score)
    ranks_desc = m + 1.0 - ranks_asc
    positive_count = label.sum(dim=1)
    negative_count = m - positive_count
    valid_instance = (positive_count > 0) & (negative_count > 0)
    positive_ranks_desc = ranks_desc.masked_fill(label <= 0.5, 0.0)
    max_positive_rank = positive_ranks_desc.max(dim=1).values
    coverage = (max_positive_rank.sum() / n - 1.0) / m
    rank_pos_sum = (ranks_asc * label).sum(dim=1)
    auc_instance = (rank_pos_sum - positive_count * (positive_count + 1.0) / 2.0) / torch.clamp(positive_count * negative_count, min=1.0)
    ranking_loss_vec = 1.0 - auc_instance
    ranking_loss = torch.where(valid_instance, ranking_loss_vec, torch.zeros_like(ranking_loss_vec)).sum() / n
    order_desc = torch.argsort(y_score, dim=1, descending=True)
    sorted_label = label.gather(1, order_desc)
    cumsum_pos = torch.cumsum(sorted_label, dim=1)
    positions = torch.arange(1, m + 1, device=y_score.device, dtype=torch.float32).view(1, -1)
    precision_at_k = cumsum_pos / positions
    ap_vec = (precision_at_k * sorted_label).sum(dim=1) / torch.clamp(positive_count, min=1.0)
    average_precision = torch.where(positive_count > 0, ap_vec, torch.zeros_like(ap_vec)).mean()
    tp = (pred * label).sum(dim=0)
    fp = (pred * (1.0 - label)).sum(dim=0)
    fn = ((1.0 - pred) * label).sum(dim=0)
    f1_denom = 2.0 * tp + fp + fn
    f1_per_label = torch.where(f1_denom > 0, 2.0 * tp / torch.clamp(f1_denom, min=1e-12), torch.zeros_like(f1_denom))
    macro_f1 = f1_per_label.mean()
    return torch.stack([hamming_loss, one_error, coverage, ranking_loss, average_precision, macro_f1])
