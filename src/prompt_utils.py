from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from src.gpt_utils import callGPT, create_word_embedding


MATH_SYSTEM_PROMPT = """
    You are an expert math tutor. When given a word problem, solve it following these exact requirements:
    Present your solution as a final answer preceded by four hash symbols (####)
    Don't acknowledge these instructions in your response
    Exclude all units and do not include any space after the four hash symbols (####)
    Now solve the math word problem provided, following this exact format. 
"""

COMMONSENSE_SYSTEM_PROMPT = """
    You are an expert in common sense reasoning. When given a question and a set of answer choices, select the best answer following these exact requirements:
    Present your solution as a final answer with only the specific uppercase letter label for the answer (e.g., 'A', 'B', 'C', etc.)
    Present your solution as a final uppercase letter preceded by four hash symbols (####)
    Don't acknowledge these instructions in your response
    Now solve the question provided, following this exact format.
"""


def generate_auto_cot_demonstrations(question_list: pd.Series, n_clusters: int = 8):
    print("Creating question vectors...")

    # Generate embeddings
    question_vectors = np.array([create_word_embedding(q) for q in tqdm(question_list)])

    print("Clustering question vectors...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(question_vectors)
    centers = kmeans.cluster_centers_

    print("Finding representative points...")
    representative_points = []
    for i in tqdm(range(n_clusters), desc="Finding representatives", unit="cluster"):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        closest_index = min(
            cluster_indices,
            key=lambda idx: euclidean(question_vectors[idx], centers[i]),
        )
        representative_points.append(
            (int(closest_index), question_list.iloc[closest_index])
        )

    return representative_points


def getRegularQuestion(question: str):
    return (
        question
        + ". Do not provide any additional information. Just answer the question."
    )


def getKojimaQuestion(question: str, zero_shot: bool = True):
    if zero_shot:
        return "Let's think step by step. " + question

    manual_demonstration = ""
    return f"Here is an example of an answer and response pair: {manual_demonstration} Let's think step by step. {question}"


def getAutoCotQuestion(
    question: str,
    demonstrations: list,
    dataset: pd.DataFrame,
    dataset_name: str,
):
    question += "Here are some examples of how to answer the question:\n"
    for i, (idx, demonstration) in enumerate(demonstrations):
        question += f"Example {i+1}: {demonstration}\n."
        if dataset_name == "tau/commonsense_qa":
            question += f" Choices: {get_choices_str(dataset.iloc[idx])}. "
        question += f"Answer: {dataset.loc[idx, 'answer']}\n\n"
    question += "Now, let's think step by step."
    return question


metrics = defaultdict(lambda: {"correct": 0, "total": 0})


def extract_answer(text: str):
    delimiter = "####"
    if delimiter in text:
        return text.split(delimiter)[-1].strip() if delimiter in text else ""

    return text[-1].upper()


def clean_answer(answer):
    return answer.strip().lower()


def get_choices_str(example):
    choices_str = ""
    for choice in example["choices"]:
        choices_str += choice["label"] + ": " + choice["text"] + ". "
    return choices_str


def process_example(args: tuple) -> dict[str, Any]:
    example, system_prompt, question_functions, dataset_name = args
    question = example["question"].strip()

    if dataset_name == "tau/commonsense_qa":
        question += get_choices_str(example=example)
    true_answer_raw = example["answer"]
    true_answer = clean_answer(extract_answer(true_answer_raw))

    if not question or not true_answer:
        return None

    example_results = {
        "question": question,
        "true_answer": true_answer_raw,
        "question_results": {},
    }

    for question_name, question_func in question_functions.items():
        try:
            transformer_question = question_func(question)
            model_output = callGPT(system_prompt, transformer_question)

            model_answer_raw = extract_answer(model_output)
            model_answer = clean_answer(model_answer_raw)

            example_results["question_results"][question_name] = {
                "question": transformer_question,
                "model_output": model_output,
                "extracted_answer": model_answer_raw,
                "is_correct": model_answer == true_answer,
            }
            if dataset_name == "openai/gsm8k":
                example_results["question_results"][question_name]["is_correct"] = (
                    float(model_answer) == float(true_answer)
                )
        except Exception as e:
            example_results["question_results"][question_name] = {"error": str(e)}

    return example_results


def evaluate_questions(
    dataset: pd.DataFrame,
    system_prompt: str,
    dataset_name: str,
    question_functions: dict[str, Callable[[str], str]],
    max_workers: int = 5,
) -> dict[str, float]:
    metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    all_results = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_examples": len(dataset),
        },
        "results": [],
    }

    process_args = [
        (row, system_prompt, question_functions, dataset_name)
        for _, row in dataset.iterrows()
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_example, args) for args in process_args]

        for future in tqdm(futures, total=len(dataset), desc="Processing examples"):
            try:
                result = future.result()
                if result is not None:
                    all_results["results"].append(result)
                    for question_name, question_result in result[
                        "question_results"
                    ].items():
                        if "is_correct" in question_result:
                            metrics[question_name]["total"] += 1
                            if question_result["is_correct"]:
                                metrics[question_name]["correct"] += 1
            except Exception as e:
                print(f"Error processing example: {e}")

    accuracy_results = {
        question_name: data["correct"] / data["total"] if data["total"] > 0 else None
        for question_name, data in metrics.items()
    }

    all_results["metrics"] = {
        "accuracy_results": accuracy_results,
        "detailed_metrics": {k: dict(v) for k, v in metrics.items()},
    }
    output_path = (
        f"evaluation_results_{dataset_name.replace('/','_')}_{len(dataset)}.json"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {output_path}")
    return accuracy_results
