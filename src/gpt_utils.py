from typing import Optional
from openai import OpenAI
import openai


def initialize_openai_model():
    client = OpenAI()
    return client


def callGPT(
    systemPrompt: str,
    question: str,
    temperature: float = 0.2,
    client: OpenAI = OpenAI(),
):
    try:
        answer = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": question},
            ],
            temperature=temperature,
        )
        return answer.choices[0].message.content
    except Exception as e:
        print(f"Error in GPT call: {e}")
        return None


def create_word_embedding(
    input: str,
    client: OpenAI = OpenAI(),
    model: Optional[str] = "text-embedding-3-small",
):
    """
    Create a word embedding for the questions. To be used in the Auto CoT method.
    """
    response = (
        client.embeddings.create(input=[input], model=model, dimensions=100)
        .data[0]
        .embedding
    )
    return response
