import json
import logging

import pytest
import responses
from pydantic import ValidationError

from finance_agent.agent import FinanceAgent
from finance_agent.classifier import ExpenseClassification
from finance_agent.exceptions import MalformedArgumentsError, MissingToolCallError

logger = logging.getLogger(__name__)


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
    def test_classify_expense_parses_valid_response(self, mock_response_food, expected_model):

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

    @responses.activate
    def test_classify_expense_raises_on_malformed_arguments(
        self, mock_response_malformed_arguments, expected_model
    ):
        responses.add(
            responses.POST,
            "http://localhost:1234/v1/chat/completions",
            json=mock_response_malformed_arguments,
            status=200,
        )
        agent = FinanceAgent(model=expected_model)

        with pytest.raises(MalformedArgumentsError):
            agent.classify_expense("Compré comida en el supermercado")

    @responses.activate
    def test_classify_expense_logs_request_details(
        self, caplog, mock_response_food, expected_model
    ):
        responses.add(
            responses.POST,
            "http://localhost:1234/v1/chat/completions",
            json=mock_response_food,
            status=200,
        )

        agent = FinanceAgent(model=expected_model)

        with caplog.at_level(logging.INFO):
            agent.classify_expense("Compré comida en el supermercado")

        assert expected_model in caplog.text
        assert "Compré comida en el supermercado" in caplog.text
        assert "tool_calls" in caplog.text

    @responses.activate
    def test_classify_expense_retry_on_invalid_category(
        self, mock_response_invalid_category, mock_response_food, expected_model
    ):
        responses.add(
            responses.POST,
            "http://localhost:1234/v1/chat/completions",
            json=mock_response_invalid_category,
            status=200,
        )

        responses.add(
            responses.POST,
            "http://localhost:1234/v1/chat/completions",
            json=mock_response_food,
            status=200,
        )

        agent = FinanceAgent(model=expected_model)
        result = agent.classify_expense("Compré comida en el supermercado")

        assert result.category == "food"
        assert len(responses.calls) == 2

    @responses.activate
    def test_classify_expense_retry_exceeds_max_attempts(
        self,
        mock_response_invalid_category,
        expected_model,
    ):
        responses.add(
            responses.POST,
            "http://localhost:1234/v1/chat/completions",
            json=mock_response_invalid_category,
            status=200,
        )
        responses.add(
            responses.POST,
            "http://localhost:1234/v1/chat/completions",
            json=mock_response_invalid_category,
            status=200,
        )

        agent = FinanceAgent(model=expected_model)

        with pytest.raises(ValidationError):
            agent.classify_expense("Compré comida en el supermercado")

        assert len(responses.calls) == 2
