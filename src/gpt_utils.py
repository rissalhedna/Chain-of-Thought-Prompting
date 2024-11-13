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