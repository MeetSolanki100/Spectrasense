import requests
import json

class Testing:
    def generate_response(self, prompt, max_tokens=150):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral:7b", "prompt":prompt, "options": {"num_predict": max_tokens, "temperature": 0.7}}
        )

        output = []
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    output.append(data["response"])
        return (''.join(output))
