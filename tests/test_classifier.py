class TestExpenseClassification:
    def test_valid_expense_classification(self):
        from finance_agent.classifier import ExpenseClassification

        # Arrange
        expense = {
            "category": "food",
            "confidence": 0.9,
            "reasoning": "personal",
        }

        # Act
        classifier = ExpenseClassification()
        result = classifier.classify(expense)

        # Assert
        assert result.category == "food"
        assert result.confidence == 0.9
        assert result.reasoning == "personal"
