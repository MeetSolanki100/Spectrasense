import requests
import json

# def generate_response(prompt, max_tokens=150):
#         response = requests.post(
#             "http://localhost:11434/api/generate",
#             json={"model": "llama3.1:8b", "prompt":prompt,"stream":True, "options": {"num_predict": max_tokens, "temperature": 0.7}}
#         )

#         output = []
#         for line in response.iter_lines():
#             if line:
#                 data = json.loads(line.decode("utf-8"))
#                 if "response" in data:
#                     output.append(data["response"])
#         return (''.join(output))
import time

# In LocalLLM.py - Fix streaming to yield tokens immediately
def generate_response(prompt, max_tokens=150):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1:8b",
        "prompt": prompt,
        "stream": True,  # Enable streaming
        'system': "You are a concise assistant. Always answer in the shortest possible way. Use only one word, number, or a very short phrase if needed. Do not explain unless explicitly asked. No filler text, no extra sentences.",
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7
        }
    }
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    yield data["response"]  # Yield immediately
                    
    except Exception as e:
        yield f"Error: {str(e)}"


print("Response from LLM:")
start_time = time.time()
full_response = []
for token in generate_response(prompt="who are you", max_tokens=50):
    print(token, end='', flush=True)  # Print immediately and stay on same line
    full_response.append(token)
end_time = time.time()
print(f"\n\nTime taken: {end_time - start_time:.2f} seconds")
    
