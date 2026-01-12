import requests
from dataclasses import dataclass

@dataclass
class OllamaClient:
    model: str = "qwen2.5:3b-instruct"
    host: str = "http://127.0.0.1:11434"

    def complete(self, prompt: str, temperature: float = 0.2, max_tokens : int = 180) -> str:
        r = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens}
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["response"]