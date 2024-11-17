from dataclasses import dataclass, field
from typing import Callable, Optional
from datasets import load_dataset
import pandas as pd

@dataclass
class CoTDataModule:
    dataset_name: str
    config_name: Optional[str] = None
    split: Optional[str] = 'train'
    cache_dir: str = '../data'
    num_samples: Optional[int] = None
    data: Optional[pd.DataFrame] = None
    processors: dict[str, Callable[[pd.DataFrame], None]] = field(init=False, default_factory=dict)
    
    def __post_init__(self):
        self.processors = {
            'tau/commonsense_qa': self._process_commonsense_qa,
            'openai/gsm8k': self._process_gsm8k
        }
        self.load_data()
        self.length = len(self.data)

    def load_data(self):
        if self.dataset_name not in self.processors:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")
        
        try:
            # Load dataset using Hugging Face datasets
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
            
            # Convert to Pandas DataFrame
            self.data = pd.DataFrame(dataset_dict)
            print(f"Loaded dataset '{self.dataset_name}' as DataFrame.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{self.dataset_name}': {e}")
        
        if self.num_samples:
            if self.num_samples > len(self.data):
                raise ValueError(f"Requested {self.num_samples} samples, but the dataset only has {len(self.data)} samples.")
            self.data = self.data.sample(n=self.num_samples, random_state=42).reset_index(drop=True)
            print(f"Sampled {self.num_samples} examples from '{self.dataset_name}'.")

        self.processors[self.dataset_name]()
    
    def _process_commonsense_qa(self):
        self.data = self.data.apply(lambda row: {
            'question': row['question'],
            'choices': [{'label': lbl, 'text': txt} for lbl, txt in zip(row['choices']['label'], row['choices']['text'])],
            'answer': row['answerKey']
        }, axis=1).to_list()
        
        self.data = pd.DataFrame(self.data)
        print("Processed 'tau/commonsense_qa' dataset.")
    
    def _process_gsm8k(self):
        # Add any specific processing if required for GSM8K dataset
        print("Processed 'openai/gsm8k' dataset.")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.length

    def __getitem__(self, index):
        """
        Retrieve a specific row or a range of rows by index.

        Args:
        - index (int or slice): Row index to retrieve, or slice for multiple rows.

        Returns:
        - pd.Series or pd.DataFrame: Row or rows as Pandas structures.
        """
        if isinstance(index, int):
            if index >= self.length or index < 0:
                raise IndexError("Index out of range")
            return self.data.iloc[index]
        
        elif isinstance(index, slice):
            return self.data.iloc[index]

        else:
            raise TypeError(f"Invalid argument type: {type(index)}. Expected int or slice.")
