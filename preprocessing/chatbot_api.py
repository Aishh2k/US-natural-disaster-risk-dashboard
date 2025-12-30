import os
import json
import requests

def get_openai_response(message, context, api_key=None):
    """
    Sends a message and context to Groq API and returns the response.
    """
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
        
    if not api_key:
        return {"error": "Groq API key not found. Please set the GROQ_API_KEY environment variable."}

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    system_prompt = """You are an AI assistant for a US Disaster Dashboard. 
    Your goal is to help users understand the 2025 disaster projections based on the provided data context.
    
    Rules:
    1. Answer questions based on the provided context.
    2. If specific data for a state is missing, provide general context about the region or common risks instead of saying "I don't have that information."
    3. Be concise, positive, and helpful.
    4. Format your response with simple HTML tags if needed (e.g., <b>bold</b>, <ul><li>lists</li></ul>).
    """

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nUser Question: {message}"}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return {"response": data['choices'][0]['message']['content']}
    except requests.exceptions.RequestException as e:
        print(f"Groq API Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
             return {"error": f"Groq API Error: {e.response.text}"}
        return {"error": str(e)}
    except Exception as e:
        print(f"General Error: {e}")
        return {"error": str(e)}
