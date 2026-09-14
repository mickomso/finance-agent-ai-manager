import json

import pytest

from finance_agent.classifier import ExpenseClassification


@pytest.fixture
def expected_model():
    return "google/gemma-4-12b-qat"


@pytest.fixture
def expected_payload(expected_model):
    return {
        "model": expected_model,
        "messages": [{"role": "user", "content": "Compré comida en el supermercado"}],
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


@pytest.fixture
def mock_response_transport():
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "ExpenseClassification",
                                "arguments": json.dumps(
                                    {
                                        "category": "transportation",
                                        "confidence": 0.8,
                                        "reasoning": "business",
                                    }
                                ),
                            }
                        }
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


@pytest.fixture
def mock_response_food():
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "ExpenseClassification",
                                "arguments": json.dumps(
                                    {
                                        "category": "food",
                                        "confidence": 0.8,
                                        "reasoning": "business",
                                    }
                                ),
                            }
                        }
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
