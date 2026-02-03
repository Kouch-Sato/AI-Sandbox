from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "あなたは親切なエージェントです"},
        {"role": "user", "content": "韓信ってカッコよくない？"}
    ]
)   

print(response.choices[0].message.content)