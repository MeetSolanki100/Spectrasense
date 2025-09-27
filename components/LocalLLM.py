import requests
import json

class LocalLLM:
    def __init__(self, model_name="mistral:7b", base_url="http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.base_url = base_url
    
    def generate_response(self, prompt, max_tokens=150):
        """Generate response using local LLM (streaming)"""
        url = self.base_url
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            output = []
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        output.append(data["response"])
            return ''.join(output).strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"