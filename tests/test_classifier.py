import pytest
from pydantic import ValidationError

from finance_agent.classifier import ExpenseClassification


class TestExpenseClassification:
    def test_valid_expense_classification(self):

        classifier = ExpenseClassification(
            category="food", confidence=0.9, reasoning="personal"
        )

        assert classifier.category == "food"
        assert classifier.confidence == 0.9
        assert classifier.reasoning == "personal"

    def test_classification_rejects_invalid_category(self):
        with pytest.raises(ValidationError):
            ExpenseClassification(
                category="invalid_category", confidence=0.9, reasoning="personal"
            )
