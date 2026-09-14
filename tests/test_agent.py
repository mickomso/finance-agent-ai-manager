import json

import pytest
import responses

from finance_agent.agent import FinanceAgent
from finance_agent.classifier import ExpenseClassification
from finance_agent.exceptions import MissingToolCallError


class TestAgent:

    @responses.activate
    def test_classify_expense_sends_correct_payload(
        self, mock_response_transport, expected_model, expected_payload
    ):

        responses.add(
            responses.POST,
            "http://localhost:1234/v1/chat/completions",
            json=mock_response_transport,
            status=200,
        )

        agent = FinanceAgent(model=expected_model)
        agent.classify_expense("Compré comida en el supermercado")

        request_data = json.loads(responses.calls[0].request.body)
        assert request_data["model"] == expected_model
        assert request_data == expected_payload

    @responses.activate
    def test_classify_expense_parses_valid_response(
        self, mock_response_food, expected_model
    ):

        responses.add(
            responses.POST,
            "http://localhost:1234/v1/chat/completions",
            json=mock_response_food,
            status=200,
        )

        agent = FinanceAgent(model=expected_model)
        result = agent.classify_expense("Compré comida en el supermercado")

        assert isinstance(result, ExpenseClassification)
        assert result.category == "food"
        assert result.confidence == 0.8
        assert result.reasoning == "business"

    @responses.activate
    def test_classify_expense_raises_when_no_tool_call(
        self, mock_response_no_tool_call, expected_model
    ):
        responses.add(
            responses.POST,
            "http://localhost:1234/v1/chat/completions",
            json=mock_response_no_tool_call,
            status=200,
        )
        agent = FinanceAgent(model=expected_model)

        with pytest.raises(MissingToolCallError):
            agent.classify_expense("Compré comida en el supermercado")

    def test_classify_expense_raises_on_malformed_arguments(self):
        pass
