import requests

def generate_completion(prompt):
    base_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "phi3",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_thread": 8,
            "num_ctx": 2024
        }
    }

    response = requests.post(base_url, json=payload)
    return response.json()

# Example usage
prompt = "Write me a Haiku about Python packaging"
result = generate_completion(prompt)
print(result)
