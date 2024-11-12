from openai import OpenAI

def initialize_openai_model():
    client = OpenAI()
    return client

def callGPT(prompt: str, client):
    answer = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    
    return answer.choices[0].message.content