import json

import pytest
import responses

from finance_agent.agent import FinanceAgent
from finance_agent.classifier import ExpenseClassification


class TestAgent:

    @pytest.fixture
    def mock_response(self):
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

    @responses.activate
    def test_classify_expense_sends_correct_payload(self, mock_response):

        responses.add(
            responses.POST,
            "http://localhost:1234/v1/chat/completions",
            json=mock_response,
            status=200,
        )

        agent = FinanceAgent(model="google/gemma-4-12b-qat")
        agent.classify_expense("Compré comida en el supermercado")

        request_data = json.loads(responses.calls[0].request.body)
        assert request_data["model"] == "google/gemma-4-12b-qat"
        assert (
            request_data["messages"][0]["content"] == "Compré comida en el supermercado"
        )
        assert request_data["messages"][0]["role"] == "user"
        assert request_data["tools"][0]["type"] == "function"
        assert request_data["tools"][0]["function"]["name"] == "ExpenseClassification"

    @responses.activate
    def test_classify_expense_parses_valid_response(self, mock_response):

        responses.add(
            responses.POST,
            "http://localhost:1234/v1/chat/completions",
            json=mock_response,
            status=200,
        )

        agent = FinanceAgent(model="google/gemma-4-12b-qat")
        result = agent.classify_expense("Compré comida en el supermercado")

        assert isinstance(result, ExpenseClassification)
        assert result.category == "transportation"
        assert result.confidence == 0.8
        assert result.reasoning == "business"

    def test_classify_expense_raises_when_no_tool_call(self):
        pass

    def test_classify_expense_raises_on_malformed_arguments(self):
        pass
