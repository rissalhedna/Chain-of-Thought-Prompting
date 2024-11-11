from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from datasets import load_dataset, Dataset

@dataclass
class CoTDataset:
    dataset_name: str
    config_name: Optional[str] = None
    split: Optional[str] = 'train'
    cache_dir: str = '../data'
    num_samples: Optional[int] = None
    data: Dataset = field(init=False)
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
        self.processors[self.dataset_name](self.data)
    
    def _process_commonsense_qa(self, dataset: Dataset):
        def join_choices(choices):
            if isinstance(choices, list):
                return '; '.join(choices)
            return choices
        self.data = self.data.map(lambda x: {'choices_str': join_choices(x['choices'])})
        print("Processed 'tau/commonsense_qa' dataset.")
    
    def _process_gsm8k(self, dataset: Dataset):
        self.data = self.data.map(lambda x: {'question_length': len(x['question'])})
        print("Processed 'openai/gsm8k' dataset.")
    
    def __getitem__(self, idx: int) -> Any:
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of bounds")
        row = self.data[idx]
        if self.dataset_name == 'tau/commonsense_qa':
            return {
                'id': row.get('id'),
                'question': row.get('question'),
                'question_concept': row.get('question_concept'),
                'choices': row.get('choices'),
                'choices_str': row.get('choices_str'),
                'answerKey': row.get('answerKey')
            }
        elif self.dataset_name == 'openai/gsm8k':
            return {
                'question': row.get('question'),
                'answer': row.get('answer'),
                'question_length': row.get('question_length')
            }
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def add_dataset_processor(self, dataset_name: str, processor: Callable[[Dataset], None]):
        self.processors[dataset_name] = processor
        print(f"Added processor for dataset '{dataset_name}'.")