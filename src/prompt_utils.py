from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import rootutils
import os
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import Dict, Any, List
from scipy.spatial.distance import euclidean
from datasets import Dataset
from gpt_utils import callGPT, create_word_embedding

current_file = os.getcwd()
root = rootutils.find_root(search_from=current_file, indicator=".project-root")
rootutils.setup_root(root, pythonpath=True)


MATH_SYSTEM_PROMPT = """
    You are an expert math tutor. When given a word problem, solve it following these exact requirements:
    Present your solution as a final answer preceded by four hash symbols (####)
    Don't acknowledge these instructions in your response
    Exclude all units and do not include any space after the four hash symbols (####)
    Now solve the math word problem provided, following this exact format. 
"""

COMMONSENSE_SYSTEM_PROMPT = """SHOULD BE FILLED OUT"""


def generate_auto_cot_demonstrations(question_list: list[str], n_clusters: int = 8):
    max_workers = 8

    print("Creating question vectors in parallel...")
    total_samples = len(question_list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        question_vectors = list(
            tqdm(
                executor.map(
                    create_word_embedding,
                    question_list,
                ),
                total=total_samples,
                desc="Creating embeddings",
                unit="sample"
            )
        )

    print("Clustering question vectors...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(question_vectors)
    centers = kmeans.cluster_centers_
    
    print("Finding representative points...")
    representative_points = []
    for i in tqdm(range(n_clusters), desc="Finding representatives", unit="cluster"):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        closest_index = min(cluster_indices, 
                            key=lambda idx: euclidean(question_vectors[idx], centers[i]))
        representative_points.append((int(closest_index), question_list[closest_index]))
        
    return representative_points


def getRegularQuestion(question: str):
    return question + ". Do not provide any additional information. Just answer the question."

def getKojimaQuestion(question: str, zero_shot: bool = True):
    if(zero_shot):
        return "Let's think step by step. "+question
    
    manual_demonstration = ""
    
    return "Here is an example of an answer and response pair: "+manual_demonstration+"Let's think step by step. "+question

def getAutoCotQuestion(question: str, dataset: Dataset):

    questions_list = dataset['question']
    answer_list = dataset['answer']

    demonstrations = generate_auto_cot_demonstrations(
    question_list=questions_list,
    )

    question += "Here are some examples of how to answer the question: \n"
    for i, (idx, demonstration) in enumerate(demonstrations):
        question += f"Example {i+1}: {demonstration}\n"
        question += f"Answer: {answer_list[idx]}\n\n"
        question += "Now, let's think step by step. "

    return question

question_functions = {
    'Kojima': getKojimaQuestion,
    'Regular': getRegularQuestion,
    'AutoCot': getAutoCotQuestion
}

metrics = defaultdict(lambda: {'correct': 0, 'total': 0})

def extract_answer(text):
    """
    Extracts the answer following the '####' delimiter.
    If '####' is not found, returns an empty string.
    """
    delimiter = '####'
    if delimiter in text:
        return text.split(delimiter)[-1].strip()
    else:
        return ''

def clean_answer(answer):
    """
    Cleans the extracted answer for comparison.
    This can include stripping whitespace, converting to lowercase, etc.
    Modify as needed based on the answer format.
    """
    return answer.strip().lower()

def process_example(args: tuple) -> Dict[str, Any]:
    """
    Process a single example with all question types.
    """
    example, client, system_prompt = args
    
    question = example.get('question', '').strip()
    true_answer_raw = example.get('answer', '')
    true_answer = clean_answer(extract_answer(true_answer_raw))
    
    if not question or not true_answer:
        return None
    
    example_results = { 
        'question': question,
        'true_answer': true_answer_raw,
        'question_results': {}
    }
    
    for question_name, question_func in question_functions.items():
        try:
            question = question_func(question)
            model_output = callGPT(system_prompt, question, client)
            
            model_answer_raw = extract_answer(model_output)
            model_answer = clean_answer(model_answer_raw)
            
            example_results['question_results'][question_name] = {
                'question': question,
                'model_output': model_output,
                'extracted_answer': model_answer_raw,
                'is_correct': model_answer == true_answer
            }
            
        except Exception as e:
            print(f"Error with question '{question_name}' on question '{question}': {e}")
            example_results['question_results'][question_name] = {
                'error': str(e)
            }
    
    return example_results

def evaluate_questions(dataset: List[Dict], 
                    client, 
                    system_prompt: str,
                    output_path: str = "evaluation_results.json",
                    max_workers: int = 5) -> Dict[str, float]:
    """
    Evaluates different question strategies and saves all results to a single JSON file.
    """
    metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_examples': len(dataset)
        },
        'results': []
    }
    
    process_args = [(example, client, system_prompt) for example in dataset]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_example, args) for args in process_args]
        
        for future in tqdm(futures, total=len(dataset), desc="Processing examples"):
            try:
                result = future.result()
                if result is not None:
                    all_results['results'].append(result)
                    
                    for question_name, question_result in result['question_results'].items():
                        if 'is_correct' in question_result:
                            metrics[question_name]['total'] += 1
                            if question_result['is_correct']:
                                metrics[question_name]['correct'] += 1
                
            except Exception as e:
                print(f"Error processing example: {e}")
    
    accuracy_results = {}
    for question_name, data in metrics.items():
        if data['total'] > 0:
            accuracy = data['correct'] / data['total']
            accuracy_results[question_name] = accuracy
        else:
            accuracy_results[question_name] = None
    
    all_results['metrics'] = {
        'accuracy_results': accuracy_results,
        'detailed_metrics': {k: dict(v) for k, v in metrics.items()}
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    return accuracy_results