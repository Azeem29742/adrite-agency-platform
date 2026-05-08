import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def get_ai_response(messages):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional AI assistant for a digital agency. Give clear, business-oriented responses."
                },
                *messages  # 👈 this adds full conversation
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"