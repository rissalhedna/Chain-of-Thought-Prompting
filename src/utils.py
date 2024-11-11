from openai import OpenAI
from datasets import load_dataset

def initialize_openai_model():
    client = OpenAI()
    return client

def callGPT(prompt: str, client: OpenAI = OpenAI()):
    answer = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    
    return answer.choices[0].message.content