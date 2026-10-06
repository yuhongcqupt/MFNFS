import argparse
import csv
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.model_selection import KFold
from .config import BUILTIN_DATASETS, DATASET_ALIASES, METRIC_KEYS, METRIC_PRINT_NAMES, DatasetSpec
from .data import load_arff_dataset, load_builtin_dataset, make_synthetic_dataset, minmax_scale_torch
from .features import build_feature_combinations
from .metrics import do_metric_torch
from .model import NFS
from .train import TrainConfig, predict_in_batches, train_one_fold
from .utils import enable_fast_cuda, get_device, set_seed

def _print_metric_line(prefix: str, values: np.ndarray) -> None:
    pass

def _load_data_for_run(args: argparse.Namespace, spec: Optional[DatasetSpec]) -> Tuple[str, np.ndarray, np.ndarray]:
    if args.synthetic:
        data_np, label_np = make_synthetic_dataset(n_samples=args.synthetic_samples, n_features=args.synthetic_features, n_labels=args.number_of_labels, seed=args.seed)
        return ('synthetic', data_np, label_np)
    if spec is not None:
        data_np, label_np = load_builtin_dataset(spec, data_dir=args.data_dir, arff_label_location=args.arff_label_location)
        return (spec.name, data_np, label_np)
    data_path = args.data_path or os.path.join(args.data_dir, f'{args.data_name}.arff')
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    data_np, label_np = load_arff_dataset(data_path, args.number_of_labels, label_location=args.arff_label_location)
    return (args.data_name, data_np, label_np)

def run_single_experiment(args: argparse.Namespace, device: torch.device, spec: Optional[DatasetSpec]=None) -> Dict[str, np.ndarray]:
    set_seed(args.seed)
    data_name, data_np, label_np = _load_data_for_run(args, spec)
    data = torch.as_tensor(data_np, dtype=torch.float32, device=device)
    label = torch.as_tensor(label_np, dtype=torch.float32, device=device)
    data = minmax_scale_torch(data)
    number_of_features = data.shape[1]
    number_of_labels = label.shape[1]
    metric_record = torch.zeros((args.folds, len(METRIC_KEYS)), device=device, dtype=torch.float32)
    if args.folds > data.shape[0]:
        raise ValueError('folds')
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    indices_np = np.arange(data.shape[0])
    expert_count_record: List[int] = []
    train_cfg = TrainConfig(epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, amp=args.amp, verbose=args.verbose)
    full_unique_indices: Optional[torch.Tensor] = None
    if args.feature_select_scope == 'full':
        full_unique_indices = build_feature_combinations(data, fs_feature_number=args.fs_feature_number, y=label, method=args.feature_select_method, bins=args.su_bins, label_weight=args.feature_label_weight, verbose=args.verbose).to(device)
    for fold_index, (train_index_np, test_index_np) in enumerate(kfold.split(indices_np)):
        train_index = torch.as_tensor(train_index_np, dtype=torch.long, device=device)
        test_index = torch.as_tensor(test_index_np, dtype=torch.long, device=device)
        train_data = data.index_select(0, train_index)
        test_data = data.index_select(0, test_index)
        train_label = label.index_select(0, train_index)
        test_label = label.index_select(0, test_index)
        if args.feature_select_scope == 'train':
            unique_indices = build_feature_combinations(train_data, fs_feature_number=args.fs_feature_number, y=train_label, method=args.feature_select_method, bins=args.su_bins, label_weight=args.feature_label_weight, verbose=args.verbose).to(device)
        else:
            assert full_unique_indices is not None
            unique_indices = full_unique_indices
        com_num = int(unique_indices.shape[0])
        expert_count_record.append(com_num)
        model = NFS(fs_number=args.fs_number, fs_feature_number=args.fs_feature_number, number_of_labels=number_of_labels, number_of_features=number_of_features, unique_indices=unique_indices, hidden_dim=args.hidden_dim, defuzzify_eps=args.defuzzify_eps).to(device)
        train_one_fold(model, train_data, train_label, train_cfg)
        y_score = predict_in_batches(model, test_data, args.eval_batch_size, args.amp)
        y_score_for_metric = torch.sigmoid(y_score) if args.metric_sigmoid else y_score
        metric_record[fold_index] = do_metric_torch(y_score_for_metric, test_label, args.threshold)
        del model, train_data, test_data, train_label, test_label, y_score, y_score_for_metric
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    metric_np = metric_record.detach().cpu().numpy()
    mean_value = metric_np.mean(axis=0)
    std_value = metric_np.std(axis=0)
    com_np = np.asarray(expert_count_record, dtype=np.int64)
    return {'dataset': data_name, 'fold_metrics': metric_np, 'mean': mean_value, 'std': std_value, 'samples': np.array([data.shape[0]], dtype=np.int64), 'features': np.array([number_of_features], dtype=np.int64), 'labels': np.array([number_of_labels], dtype=np.int64), 'com_num': np.array([float(com_np.mean())], dtype=np.float64), 'com_num_std': np.array([float(com_np.std())], dtype=np.float64), 'fold_com_nums': com_np, 'model_version': args.model_version, 'feature_select_method': args.feature_select_method, 'feature_label_weight': np.array([args.feature_label_weight], dtype=np.float64), 'feature_select_scope': args.feature_select_scope, 'defuzzify_mode': 'normalized', 'defuzzify_eps': np.array([args.defuzzify_eps], dtype=np.float64)}

def _normalize_dataset_name(name: str) -> str:
    if name in BUILTIN_DATASETS:
        return name
    key = name.lower()
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    raise ValueError(name)

def _selected_builtin_specs(args: argparse.Namespace) -> List[DatasetSpec]:
    if args.datasets:
        names = [_normalize_dataset_name(name) for name in args.datasets]
        return [BUILTIN_DATASETS[name] for name in names]
    if args.data_name.lower() == 'all':
        return list(BUILTIN_DATASETS.values())
    name = _normalize_dataset_name(args.data_name)
    return [BUILTIN_DATASETS[name]]

def save_results(results: List[Dict[str, np.ndarray]], args: argparse.Namespace) -> Tuple[str, str, str]:
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    prefix = os.path.join(args.output_dir, f'mfnfs_final_norm_alpha01_results_{timestamp}')
    summary_csv_path = prefix + '_summary.csv'
    fold_csv_path = prefix + '_folds.csv'
    txt_path = prefix + '.txt'
    summary_header = ['dataset', 'model_version', 'samples', 'features', 'labels', 'com_num_mean', 'com_num_std', 'epochs', 'folds', 'fs_number', 'fs_feature_number', 'learning_rate', 'batch_size', 'feature_select_method', 'feature_label_weight', 'feature_select_scope', 'defuzzify_mode', 'defuzzify_eps', 'metric_sigmoid']
    for key in METRIC_KEYS:
        summary_header.extend([f'{key}_mean', f'{key}_std'])
    with open(summary_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(summary_header)
        for res in results:
            row = [res['dataset'], res.get('model_version', args.model_version), int(res['samples'][0]), int(res['features'][0]), int(res['labels'][0]), float(res['com_num'][0]), float(res.get('com_num_std', np.array([0.0]))[0]), args.epochs, args.folds, args.fs_number, args.fs_feature_number, args.learning_rate, args.batch_size, res.get('feature_select_method', args.feature_select_method), float(res.get('feature_label_weight', np.array([args.feature_label_weight]))[0]), res.get('feature_select_scope', args.feature_select_scope), res.get('defuzzify_mode', args.defuzzify_mode), float(res.get('defuzzify_eps', np.array([args.defuzzify_eps]))[0]), args.metric_sigmoid]
            for mean_v, std_v in zip(res['mean'], res['std']):
                row.extend([float(mean_v), float(std_v)])
            writer.writerow(row)
    with open(fold_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'model_version', 'fold', 'com_num'] + METRIC_KEYS)
        for res in results:
            fold_com_nums = res.get('fold_com_nums', np.zeros(len(res['fold_metrics']), dtype=np.int64))
            for fold_idx, fold_values in enumerate(res['fold_metrics'], start=1):
                writer.writerow([res['dataset'], res.get('model_version', args.model_version), fold_idx, int(fold_com_nums[fold_idx - 1]) if len(fold_com_nums) >= fold_idx else ''] + [float(v) for v in fold_values])
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('MFNFS final version results\n')
        f.write(f'time: {timestamp}\n')
        f.write(f'model_version: {args.model_version}\n')
        f.write(f'epochs: {args.epochs}\n')
        f.write(f'folds: {args.folds}\n')
        f.write(f'fs_number: {args.fs_number}\n')
        f.write(f'fs_feature_number: {args.fs_feature_number}\n')
        f.write(f'learning_rate: {args.learning_rate}\n')
        f.write(f'batch_size: {args.batch_size}\n')
        f.write(f'feature_select_method: {args.feature_select_method}\n')
        f.write(f'feature_label_weight: {args.feature_label_weight}\n')
        f.write(f'feature_select_scope: {args.feature_select_scope}\n')
        f.write(f'defuzzify_mode: {args.defuzzify_mode}\n')
        f.write(f'defuzzify_eps: {args.defuzzify_eps}\n')
        f.write(f'metric_sigmoid: {args.metric_sigmoid}\n')
        f.write('metrics: HL, OE, CV, RL, AP, Macro-F1\n\n')
        for res in results:
            f.write('=' * 80 + '\n')
            f.write(f"dataset: {res['dataset']}\n")
            f.write(f"model_version: {res.get('model_version', args.model_version)}\n")
            f.write(f"feature_select_method: {res.get('feature_select_method', args.feature_select_method)}\n")
            f.write(f"feature_label_weight: {float(res.get('feature_label_weight', np.array([args.feature_label_weight]))[0]):.6f}\n")
            f.write(f"feature_select_scope: {res.get('feature_select_scope', args.feature_select_scope)}\n")
            f.write(f"defuzzify_mode: {res.get('defuzzify_mode', args.defuzzify_mode)}\n")
            f.write(f"samples: {int(res['samples'][0])}, features: {int(res['features'][0])}, labels: {int(res['labels'][0])}, com_num_mean: {float(res['com_num'][0]):.2f}, com_num_std: {float(res.get('com_num_std', np.array([0.0]))[0]):.2f}\n")
            f.write('mean: ' + ', '.join([f'{k}={v:.6f}' for k, v in zip(METRIC_KEYS, res['mean'])]) + '\n')
            f.write('std:  ' + ', '.join([f'{k}={v:.6f}' for k, v in zip(METRIC_KEYS, res['std'])]) + '\n')
    return (summary_csv_path, fold_csv_path, txt_path)

def _run_selected_datasets(args: argparse.Namespace, device: torch.device) -> List[Dict[str, np.ndarray]]:
    if args.synthetic or args.data_path:
        return [run_single_experiment(args, device, spec=None)]
    specs = _selected_builtin_specs(args)
    results = []
    for spec in specs:
        results.append(run_single_experiment(args, device, spec=spec))
    return results

def run_experiment(args: argparse.Namespace) -> List[Dict[str, np.ndarray]]:
    enable_fast_cuda()
    device = get_device(args.device)
    if device.type == 'cuda':
        pass
    results = _run_selected_datasets(args, device)
    if args.save_results:
        summary_csv_path, fold_csv_path, txt_path = save_results(results, args)
    return results

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str, default='all')
    parser.add_argument('--datasets', type=str, nargs='+', default=None)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--number-of-labels', type=int, default=53)
    parser.add_argument('--arff-label-location', type=str, default='end', choices=['start', 'end'])
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--fs-number', type=int, default=2)
    parser.add_argument('--fs-feature-number', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--eval-batch-size', type=int, default=4096)
    parser.add_argument('--hidden-dim', type=int, default=500)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--feature-select-scope', type=str, default='train', choices=['train', 'full'])
    parser.add_argument('--su-bins', type=int, default=16)
    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument('--metric-sigmoid', dest='metric_sigmoid', action='store_true')
    metric_group.add_argument('--no-metric-sigmoid', dest='metric_sigmoid', action='store_false')
    parser.set_defaults(metric_sigmoid=True)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results')
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument('--save-results', dest='save_results', action='store_true')
    save_group.add_argument('--no-save-results', dest='save_results', action='store_false')
    parser.set_defaults(save_results=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--synthetic-samples', type=int, default=512)
    parser.add_argument('--synthetic-features', type=int, default=32)
    parser.set_defaults(model_version='normalized_defuzzification_feature_label_alpha_0.1', feature_select_method='su_label_hist', feature_label_weight=0.1, defuzzify_mode='normalized', defuzzify_eps=1e-12)
    return parser
