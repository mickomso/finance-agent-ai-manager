import json

import requests

from finance_agent.classifier import ExpenseClassification
from finance_agent.exceptions import MissingToolCallError


class FinanceAgent:
    def __init__(self, model: str, base_url: str = "http://localhost:1234/v1"):
        self.model = model
        self.base_url = base_url

    def classify_expense(self, text: str) -> ExpenseClassification:
        url = f"{self.base_url}/chat/completions"

        # Preparamos el payload que simula una petición de chat con llamadas a funciones (tool use)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ExpenseClassification",
                        "description": "Clasifica un gasto",
                        "parameters": ExpenseClassification.model_json_schema(),
                    },
                }
            ],
        }

        # Realizamos la petición POST que interceptará la librería 'responses'
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        try:
            tool_call = data["choices"][0]["message"]["tool_calls"][0]
            arguments_str = tool_call["function"]["arguments"]
            arguments_dict = json.loads(arguments_str)

            return ExpenseClassification(**arguments_dict)

        except (KeyError, IndexError):
            raise MissingToolCallError(
                "No se encontró la llamada a la herramienta requerida en la respuesta."
            )
