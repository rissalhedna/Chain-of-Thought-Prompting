import math
import rootutils
import matplotlib

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

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
        question_list, n_clusters=int(math.sqrt(first_k_samples) / 2)
    )

    evaluate_questions(
        dataset=dataset[:first_k_samples],
        system_prompt=system_prompt,
        question_functions=question_functions,
        max_workers=4,
        dataset_name=dataset_name,
        visualize_sample=visualize_sample,
    )


evaluate_dataset(
    dataset_name="tau/commonsense_qa",
    system_prompt=COMMONSENSE_SYSTEM_PROMPT,
    first_k_samples=5,
    visualize_sample=4,
)

# evaluate_dataset("tau/commonsense_qa", COMMONSENSE_SYSTEM_PROMPT, 1000)
