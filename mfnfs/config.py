from dataclasses import dataclass
from typing import Dict, Optional
METRIC_KEYS = ['HL', 'OE', 'CV', 'RL', 'AP', 'Macro-F1']
METRIC_PRINT_NAMES = ['hamming loss', 'one error', 'coverage', 'ranking loss', 'average precision', 'macro-F1']

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    number_of_labels: int
    filename: Optional[str] = None
    feature_filename: Optional[str] = None
    label_filename: Optional[str] = None
BUILTIN_DATASETS: Dict[str, DatasetSpec] = {'emotions': DatasetSpec(name='xxx', kind='yyy', number_of_labels=6, filename='xxx.yyy')}
DATASET_ALIASES = {name.lower(): name for name in BUILTIN_DATASETS.keys()}
