import os
from typing import Tuple
import numpy as np
import torch
from .config import DatasetSpec

def _to_dense_array(x):
    if hasattr(x, 'toarray'):
        return x.toarray()
    return np.asarray(x)

def _resolve_existing_path(data_dir: str, filename: str) -> str:
    direct_path = os.path.join(data_dir, filename)
    if os.path.exists(direct_path):
        return direct_path
    if os.path.isdir(data_dir):
        target = filename.lower()
        for entry in os.listdir(data_dir):
            if entry.lower() == target:
                return os.path.join(data_dir, entry)
    return direct_path

def _to_float32_features(x) -> np.ndarray:
    x_np = _to_dense_array(x).astype(np.float32)
    return np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)

def _to_binary_labels(y) -> np.ndarray:
    y_np = _to_dense_array(y).astype(np.float32)
    y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0)
    return (y_np > 0).astype(np.float32)

def load_arff_dataset(data_path: str, number_of_labels: int, label_location: str='end') -> Tuple[np.ndarray, np.ndarray]:

    X, y = load_from_arff(data_path, label_count=number_of_labels, label_location=label_location)
    data_np = _to_float32_features(X)
    label_np = _to_binary_labels(y)
    return (data_np, label_np)

def load_numpy_pair_dataset(feature_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data_np = _to_float32_features(np.load(feature_path))
    label_np = _to_binary_labels(np.load(label_path))
    return (data_np, label_np)

def load_csv_last_labels_dataset(data_path: str, number_of_labels: int) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    data_np = df.iloc[:, :-number_of_labels].to_numpy(dtype=np.float32)
    label_np = df.iloc[:, -number_of_labels:].to_numpy(dtype=np.float32)
    return (_to_float32_features(data_np), _to_binary_labels(label_np))

def load_builtin_dataset(spec: DatasetSpec, data_dir: str, arff_label_location: str='end') -> Tuple[np.ndarray, np.ndarray]:
    if spec.kind == 'arff':
        assert spec.filename is not None
        data_path = _resolve_existing_path(data_dir, spec.filename)
        return load_arff_dataset(data_path, spec.number_of_labels, label_location=arff_label_location)
    if spec.kind == 'npy_pair':
        assert spec.feature_filename is not None and spec.label_filename is not None
        feature_path = _resolve_existing_path(data_dir, spec.feature_filename)
        label_path = _resolve_existing_path(data_dir, spec.label_filename)
        return load_numpy_pair_dataset(feature_path, label_path)
    if spec.kind == 'csv_last_labels':
        assert spec.filename is not None
        data_path = _resolve_existing_path(data_dir, spec.filename)
        return load_csv_last_labels_dataset(data_path, spec.number_of_labels)
    raise ValueError(f'{spec.kind}')

def make_synthetic_dataset(n_samples: int=512, n_features: int=32, n_labels: int=6, seed: int=0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    w = rng.normal(size=(n_features, n_labels)).astype(np.float32)
    logits = x @ w + 0.5 * rng.normal(size=(n_samples, n_labels)).astype(np.float32)
    threshold = np.quantile(logits, 0.72, axis=0, keepdims=True)
    y = (logits >= threshold).astype(np.float32)
    empty = y.sum(axis=1) == 0
    if np.any(empty):
        y[empty, np.argmax(logits[empty], axis=1)] = 1.0
    return (x, y)

def minmax_scale_torch(x: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    x_min = torch.amin(x, dim=0, keepdim=True)
    x_max = torch.amax(x, dim=0, keepdim=True)
    denom = torch.clamp(x_max - x_min, min=eps)
    return (x - x_min) / denom
