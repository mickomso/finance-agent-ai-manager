import json

import requests

from finance_agent.classifier import ExpenseClassification
from finance_agent.exceptions import MalformedArgumentsError, MissingToolCallError


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
            message = data["choices"][0]["message"]
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                raise MissingToolCallError(
                    "El LLM no devolvió ninguna llamada a herramienta."
                )

            tool_call = tool_calls[0]
            arguments_str = tool_call["function"]["arguments"]

            # Intentamos parsear los argumentos como JSON
            try:
                arguments_dict = json.loads(arguments_str)
            except json.JSONDecodeError as e:
                raise MalformedArgumentsError(
                    "Los argumentos del tool_call no son un JSON válido."
                ) from e

        except (KeyError, IndexError) as e:
            if isinstance(e, MalformedArgumentsError):
                raise
            raise MissingToolCallError(
                "La estructura de la respuesta del LLM es inválida."
            ) from e

        return ExpenseClassification(**arguments_dict)
