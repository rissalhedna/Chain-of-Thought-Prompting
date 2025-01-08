import math
import rootutils
import matplotlib

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from src.batch_processing import BatchProcessor
from src.gpt_utils import callGPT
from src.data_utils import CoTDataModule
from src.prompt_utils import (
    generate_auto_cot_demonstrations,
    evaluate_questions,
    MATH_SYSTEM_PROMPT,
    COMMONSENSE_SYSTEM_PROMPT,
    getAutoCotQuestion,
    getKojimaQuestion,
    getRegularQuestion,
    getTreeReasoningQuestion,
    build_reasoning_tree,
    get_majority_answer,
)
from src.batch_preparation import create_batches


def evaluate_dataset(
    dataset_name: str,
    system_prompt: str,
    first_k_samples: int,
    visualize_sample: int = 0,
):
    matplotlib.use("Agg")  # Use non-interactive backend

    def getAutoCotQuestionWrapper(question: str):
        return getAutoCotQuestion(
            question=question,
            demonstrations=auto_cot_demonstrations,
            dataset=dataset,
            dataset_name=dataset_name,
        )

    def getTreeReasoningWrapper(question: str):
        responses = []
        temperatures = [0.2, 0.4, 0.6, 0.8, 1.0]
        formatted_question = getTreeReasoningQuestion(question)

        for temp in temperatures:
            response = callGPT(system_prompt, formatted_question, temperature=temp)
            responses.append(response)

        tree = build_reasoning_tree(
            responses, question=question, save_path=f"reasoning_trees/{dataset_name}"
        )
        majority_answer = get_majority_answer(tree)

        return majority_answer

    question_functions = {
        "Kojima": getKojimaQuestion,
        "Regular": getRegularQuestion,
        "AutoCot": getAutoCotQuestionWrapper,
        "TreeReasoning": getTreeReasoningWrapper,
    }

    dataset = CoTDataModule(dataset_name).data

    question_list = dataset["question"][:first_k_samples]

    auto_cot_demonstrations = generate_auto_cot_demonstrations(
        question_list,
        n_clusters=int(math.sqrt(first_k_samples) / 2),
        # n_clusters=2,
    )

    evaluate_questions(
        dataset=dataset[:first_k_samples],
        system_prompt=system_prompt,
        question_functions=question_functions,
        max_workers=4,
        dataset_name=dataset_name,
        visualize_sample=visualize_sample,
    )


def process_dataset_in_batches(
    dataset_name: str, first_k_samples: int, batch_size: int = 100
):
    dataset = CoTDataModule(dataset_name).data[:first_k_samples]

    auto_cot_demonstrations = generate_auto_cot_demonstrations(
        dataset["question"], n_clusters=int(math.sqrt(first_k_samples) / 2)
    )

    create_batches(
        dataset=dataset,
        batch_size=batch_size,
        dataset_name=dataset_name,
        auto_cot_demonstrations=auto_cot_demonstrations,
    )


# evaluate_dataset(
#     dataset_name="openai/gsm8k",
#     system_prompt=MATH_SYSTEM_PROMPT,
#     first_k_samples=10,
#     visualize_sample=4,
# )

# evaluate_dataset(
#     dataset_name="tau/commonsense_qa",
#     system_prompt=COMMONSENSE_SYSTEM_PROMPT,
#     first_k_samples=10,
#     visualize_sample=4,
# )

# evaluate_dataset("tau/commonsense_qa", COMMONSENSE_SYSTEM_PROMPT, 1000)

process_dataset_in_batches(
    dataset_name="openai/gsm8k", first_k_samples=10, batch_size=2
)


# processor = BatchProcessor()
# processor.process_all_batches()
