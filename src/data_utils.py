from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from datasets import load_dataset
from datasets.arrow_dataset import Dataset

@dataclass
class CoTDataModule:
    dataset_name: str
    config_name: Optional[str] = None
    split: Optional[str] = 'train'
    cache_dir: str = '../data'
    num_samples: Optional[int] = None
    data: Optional[Dataset] = None
    processors: Dict[str, Callable[[Dataset], None]] = field(init=False, default_factory=dict)
    
    def __post_init__(self):
        self.processors = {
            'tau/commonsense_qa': self._process_commonsense_qa,
            'openai/gsm8k': self._process_gsm8k
        }
        self.load_data()
    
    def load_data(self):
        if self.dataset_name not in self.processors:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")
        try:
            if self.dataset_name == 'openai/gsm8k':
                dataset_dict = load_dataset(
                    self.dataset_name,
                    'main',
                    cache_dir=self.cache_dir,
                    split=self.split
                )
            else:
                dataset_dict = load_dataset(
                    self.dataset_name,
                    cache_dir=self.cache_dir,
                    split=self.split
                )
            self.data = dataset_dict
            print(f"Loaded dataset '{self.dataset_name}'.")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{self.dataset_name}': {e}")
        if self.num_samples:
            if self.num_samples > len(self.data):
                raise ValueError(f"Requested {self.num_samples} samples, but the dataset only has {len(self.data)} samples.")
            self.data = self.data.shuffle(seed=42).select(range(self.num_samples))
            print(f"Sampled {self.num_samples} examples from '{self.dataset_name}'.")

        self.processors[self.dataset_name]()
    
    def _process_commonsense_qa(self):
        def join_choices(choices):
            if isinstance(choices, list):
                return '; '.join(choices)
            return choices
        self.data = self.data.map(lambda x: {'choices_str': join_choices(x['choices'])})
        print("Processed 'tau/commonsense_qa' dataset.")
    
    def _process_gsm8k(self):
        self.data = self.data.map(lambda x: {'question_length': len(x['question'])})
        print("Processed 'openai/gsm8k' dataset.")
    
    def __len__(self) -> int:
        return len(self.data)
    