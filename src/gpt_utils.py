from typing import Optional
from openai import OpenAI

def initialize_openai_model():
    client = OpenAI()
    return client

def callGPT(systemPrompt:str, question: str, client):
    answer = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": question}],
    )
    
    return answer.choices[0].message.content

def create_word_embedding(input: str, client: OpenAI = OpenAI(), model: Optional[str]= "text-embedding-3-small"):
    """
    Create a word embedding for the questions. To be used in the Auto CoT method.
    """
    response = client.embeddings.create(
            input=[input],
            model=model,
            dimensions = 100).data[0].embedding
    return response