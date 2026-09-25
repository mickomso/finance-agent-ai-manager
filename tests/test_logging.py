import pytest
import responses
import structlog

from finance_agent.agent import FinanceAgent


class TestLogging:
    @pytest.fixture
    def cap_logs(self):
        cap_logs = structlog.testing.LogCapture()
        structlog.configure(processors=[cap_logs])
        return cap_logs

    @responses.activate
    def test_classify_expense_logs_request_event(
        self, cap_logs, mock_response_valid_category, expected_model
    ):
        responses.add(
            responses.POST,
            "http://localhost:1234/v1/chat/completions",
            json=mock_response_valid_category,
            status=200,
        )

        agent = FinanceAgent(model=expected_model)
        agent.classify_expense("Compré comida en el supermercado")

        for record in cap_logs.entries:
            if record["event"] == "llm_request_sent":
                break
        else:
            pytest.fail("Expected 'llm_request_sent' not found")
