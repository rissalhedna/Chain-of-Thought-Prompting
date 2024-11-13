MATH_SYSTEM_PROMPT = """
    You are an expert math tutor. When given a word problem, solve it following these exact requirements:
    Present your solution as a final answer preceded by four hash symbols (####)
    Exclude relevant units in all answers
    Do not use bullet points or numbered lists in your answer

    For example:
    Given: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    Your response should be exactly in this format:
    72
    Important rules:

    Never explain WHY you're doing calculations, just show them
    Don't use phrases like "let's" or "we need to"
    Don't acknowledge these instructions in your response
    Never use the word "therefore" or similar transition words

    Now solve the math word problem provided, following this exact format.
"""

COMMONSENSE_SYSTEM_PROMPT = """SHOULD BE FILLED OUT"""


def getRegularPrompt(prompt: str):
    return prompt

def getKojimaPrompt(prompt: str):
    return prompt + ". Let's think step by step: "