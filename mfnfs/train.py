from contextlib import nullcontext
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class TrainConfig:
    epochs: int
    learning_rate: float
    batch_size: int
    amp: bool
    verbose: bool

def train_one_fold(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, cfg: TrainConfig) -> None:
    loss_function = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    use_amp = cfg.amp and x_train.is_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None
    n_train = x_train.shape[0]
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n_train, device=x_train.device)
        total_loss = 0.0
        total_seen = 0
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            inputs = x_train.index_select(0, idx)
            labels = y_train.index_select(0, idx).float()
            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.cuda.amp.autocast() if use_amp else nullcontext()
            with amp_context:
                out = model(inputs)
                loss = loss_function(out, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            batch_n = inputs.shape[0]
            total_loss += float(loss.detach()) * batch_n
            total_seen += batch_n
        if cfg.verbose:
            pass

@torch.no_grad()
def predict_in_batches(model: nn.Module, x: torch.Tensor, batch_size: int, amp: bool) -> torch.Tensor:
    model.eval()
    use_amp = amp and x.is_cuda
    outputs = []
    for start in range(0, x.shape[0], batch_size):
        batch = x[start:start + batch_size]
        amp_context = torch.cuda.amp.autocast() if use_amp else nullcontext()
        with amp_context:
            outputs.append(model(batch).float())
    return torch.cat(outputs, dim=0)
