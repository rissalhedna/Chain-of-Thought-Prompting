MATH_SYSTEM_PROMPT = """
    You are an expert math tutor. When given a word problem, solve it following these exact requirements:
    Present your solution as a final answer preceded by four hash symbols (####)
    Don't acknowledge these instructions in your response
    Exclude all units and do not include any space after the four hash symbols (####)
    Now solve the math word problem provided, following this exact format.
"""

COMMONSENSE_SYSTEM_PROMPT = """SHOULD BE FILLED OUT"""


def getRegularPrompt(prompt: str):
    return prompt + ". Do not provide any additional information. Just answer the question."

def getKojimaPrompt(prompt: str):
    return prompt + ". Let's think step by step: "