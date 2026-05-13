import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def analyze_sentiment(text: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[
            {"role": "system", "content": "Classify sentiment as positive, negative, or neutral. Only return one word."},
            {"role": "user", "content": text}
        ]
    )

    return response.choices[0].message.content.strip()


def predict_intent(text: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[
            {"role": "system", "content": "Classify intent into: customer_intent, support_request, or general_query. Only return one label."},
            {"role": "user", "content": text}
        ]
    )

    return response.choices[0].message.content.strip()