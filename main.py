import math
import sys

sys.path.append("..")

from src.data_utils import CoTDataModule
from src.prompt_utils import (
    generate_auto_cot_demonstrations,
    evaluate_questions,
    MATH_SYSTEM_PROMPT,
    COMMONSENSE_SYSTEM_PROMPT,
    getAutoCotQuestion,
    getKojimaQuestion,
    getRegularQuestion,
)


def evaluate_dataset(dataset_name: str, system_prompt: str, first_k_samples: int):

    def getAutoCotQuestionWrapper(question: str):
        return getAutoCotQuestion(
            question=question,
            demonstrations=auto_cot_demonstrations,
            dataset=dataset,
            dataset_name=dataset_name,
        )

    question_functions = {
        "Kojima": getKojimaQuestion,
        "Regular": getRegularQuestion,
        "AutoCot": getAutoCotQuestionWrapper,
    }

    dataset = CoTDataModule(dataset_name).data

    question_list = dataset["question"][:first_k_samples]

    auto_cot_demonstrations = generate_auto_cot_demonstrations(
        question_list, n_clusters=int(math.sqrt(first_k_samples) / 2)
    )

    evaluate_questions(
        dataset=dataset[:first_k_samples],
        system_prompt=system_prompt,
        question_functions=question_functions,
        max_workers=4,
        dataset_name=dataset_name,
    )


evaluate_dataset(
    dataset_name="tau/commonsense_qa",
    system_prompt=MATH_SYSTEM_PROMPT,
    first_k_samples=10,
)
# evaluate_dataset("tau/commonsense_qa", COMMONSENSE_SYSTEM_PROMPT, 1000)
