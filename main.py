import sys
sys.path.append("..")

from src.prompt_utils import (
    generate_auto_cot_demonstrations,
    evaluate_questions,
    MATH_SYSTEM_PROMPT,
    getAutoCotQuestion,
    getKojimaQuestion,
    getRegularQuestion
)
from src.data_utils import CoTDataModule

math_ds = CoTDataModule("openai/gsm8k").data
commonsense_ds = CoTDataModule("tau/commonsense_qa").data

question_list = math_ds['question'][:10]
auto_cot_demonstrations = generate_auto_cot_demonstrations(question_list, n_clusters=3)

def getAutoCotQuestionWrapper(question: str):
    return getAutoCotQuestion(question=question, demonstrations=auto_cot_demonstrations, dataset=math_ds)

question_functions = {
    'Kojima': getKojimaQuestion,
    'Regular': getRegularQuestion,
    'AutoCot': getAutoCotQuestionWrapper
}

evaluate_questions(
    dataset=math_ds[:10], 
    system_prompt=MATH_SYSTEM_PROMPT, 
    question_functions=question_functions, 
    max_workers=4
)
