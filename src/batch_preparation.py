import os
import json
from pathlib import Path
from typing import Dict, List, Callable
import pandas as pd
from src.prompt_utils import (
    getKojimaQuestion,
    getRegularQuestion,
    getAutoCotQuestion,
    getTreeReasoningQuestion,
    MATH_SYSTEM_PROMPT,
    COMMONSENSE_SYSTEM_PROMPT,
)


def create_batch_structure(base_path: str = "./batch_files") -> Dict[str, str]:
    """Create folder structure for different prompting methods."""
    method_paths = {
        "kojima": "kojima_batches",
        "regular": "regular_batches",
        "auto_cot": "auto_cot_batches",
        "tree_reasoning": "tree_reasoning_batches",
    }

    for method_folder in method_paths.values():
        full_path = Path(base_path) / method_folder
        full_path.mkdir(parents=True, exist_ok=True)

    return {k: str(Path(base_path) / v) for k, v in method_paths.items()}


def create_original_batches(
    dataset: pd.DataFrame,
    batch_size: int,
    base_path: str = "./batch_files",
) -> None:
    """Create batches of the original dataset with custom IDs."""
    original_data_path = Path("original_batches")
    original_data_path.mkdir(parents=True, exist_ok=True)

    # Calculate number of batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    # Process each batch
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_data = dataset.iloc[start_idx:end_idx].copy()

        # Add custom_id to each row
        batch_data["custom_id"] = batch_data.index.map(lambda x: f"question_{x}")

        # Save batch
        batch_file = original_data_path / f"batch_{batch_num}.jsonl"
        with open(batch_file, "w") as f:
            for _, row in batch_data.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")


def prepare_prompt_payload(
    row: pd.Series,
    method: str,
    system_prompt: str,
    dataset_name: str,
    dataset: pd.DataFrame,
    auto_cot_demonstrations: List = None,
) -> Dict:
    """Prepare the prompt payload for a specific method."""
    question = row["question"]

    # Add choices for commonsense_qa dataset
    if dataset_name == "tau/commonsense_qa":
        choices_str = ""
        for choice in row["choices"]:
            choices_str += f"{choice['label']}: {choice['text']}. "
        question += choices_str

    # Format question based on method
    if method == "kojima":
        formatted_question = getKojimaQuestion(question)
    elif method == "regular":
        formatted_question = getRegularQuestion(question)
    elif method == "auto_cot":
        formatted_question = getAutoCotQuestion(
            question=question,
            demonstrations=auto_cot_demonstrations,
            dataset=dataset,
            dataset_name=dataset_name,
        )
    else:  # tree_reasoning
        formatted_question = getTreeReasoningQuestion(question)

    return {
        "custom_id": row["custom_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_question},
            ],
            "temperature": 0.2,
        },
    }


def create_batches(
    dataset: pd.DataFrame,
    batch_size: int,
    dataset_name: str,
    auto_cot_demonstrations: List = None,
    base_path: str = "./batch_files",
) -> None:
    """Create batches for all prompting methods."""
    # First, create original batches with custom IDs
    create_original_batches(dataset, batch_size, base_path)

    # Create folder structure for formatted batches
    method_paths = create_batch_structure(base_path)

    # Determine system prompt based on dataset
    system_prompt = (
        MATH_SYSTEM_PROMPT
        if dataset_name == "openai/gsm8k"
        else COMMONSENSE_SYSTEM_PROMPT
    )

    # Calculate number of batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    # Process each batch
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_data = dataset.iloc[start_idx:end_idx].copy()

        # Add custom_id to batch_data
        batch_data["custom_id"] = batch_data.index.map(lambda x: f"question_{x}")

        # Create batch files for each method
        for method, method_path in method_paths.items():
            batch_file = Path(method_path) / f"batch_{batch_num}.jsonl"

            with open(batch_file, "w") as f:
                for _, row in batch_data.iterrows():
                    payload = prepare_prompt_payload(
                        row=row,
                        method=method,
                        system_prompt=system_prompt,
                        dataset_name=dataset_name,
                        dataset=dataset,
                        auto_cot_demonstrations=auto_cot_demonstrations,
                    )
                    f.write(json.dumps(payload) + "\n")

            print(f"Created batch {batch_num} for method {method}")
