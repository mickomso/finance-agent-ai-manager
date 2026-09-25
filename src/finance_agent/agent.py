import json

import requests
import structlog
from pydantic import ValidationError

from finance_agent.classifier import ExpenseClassification
from finance_agent.exceptions import MalformedArgumentsError, MissingToolCallError


def configure_structlog() -> None:
    """Route structlog through stdlib logging so handlers like pytest's caplog work."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.KeyValueRenderer(key_order=["event"]),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )


configure_structlog()

logger = structlog.get_logger(__name__)


def retry_on_validation_error(max_attempts=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                with structlog.contextvars.bound_contextvars(attempt=attempt):
                    if attempt == 0:
                        text = args[1]
                    else:
                        text = (
                            "Classify using ONLY: food, transportation, entertainment, "
                            f"utilities, other. Expense: {args[1]}"
                        )
                    try:
                        return func(args[0], text)
                    except ValidationError:
                        logger.warning("invalid_category_returned", attempt=attempt)
                        if attempt == max_attempts - 1:
                            raise

        return wrapper

    return decorator


class FinanceAgent:
    def __init__(self, model: str, base_url: str = "http://localhost:1234/v1"):
        self.model = model
        self.base_url = base_url

    @retry_on_validation_error(max_attempts=2)
    def classify_expense(self, text: str) -> ExpenseClassification:
        url = f"{self.base_url}/chat/completions"

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
        # attempt se añade automáticamente por el contexto del decorator
        logger.info("llm_request_sent", model=self.model, url=url, payload=payload)
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        logger.info("llm_response_received", data=data)

        try:
            message = data["choices"][0]["message"]
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                raise MissingToolCallError(
                    "El LLM no devolvió ninguna llamada a herramienta."
                )

            tool_call = tool_calls[0]
            arguments_str = tool_call["function"]["arguments"]

            try:
                arguments_dict = json.loads(arguments_str)
            except json.JSONDecodeError as e:
                raise MalformedArgumentsError(
                    "Los argumentos del tool_call no son un JSON válido."
                ) from e

        except (KeyError, IndexError, MalformedArgumentsError, ValidationError) as e:
            if isinstance(e, (MalformedArgumentsError, ValidationError)):
                raise
            raise MissingToolCallError(
                "La estructura de la respuesta del LLM es inválida."
            ) from e

        return ExpenseClassification(**arguments_dict)
