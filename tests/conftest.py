import json

import pytest

from finance_agent.agent import configure_structlog
from finance_agent.classifier import ExpenseClassification


@pytest.fixture(autouse=True)
def _reset_structlog():
    """Reinstate the app's structlog config before each test.

    Tests that reconfigure structlog for their own purposes (e.g. to
    capture log entries) would otherwise leak that config into whichever
    test runs next.
    """
    configure_structlog()


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


@pytest.fixture
def mock_response_groceries():
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
                                        "category": "groceries",
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
def mock_response_valid_category():
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


@pytest.fixture
def mock_response_invalid_category():
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
                                        "category": "invalid_category",
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
def mock_response_no_tool_call():
    return {
        "choices": [{"message": {"tool_calls": []}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


@pytest.fixture
def mock_response_malformed_arguments():
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "ExpenseClassification",
                                "arguments": "{malformed_json}",
                            }
                        }
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
