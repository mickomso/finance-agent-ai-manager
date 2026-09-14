class ExpenseClassification:
    def classify(self, expense):
        class Result:
            def __init__(self, category, confidence, reasoning):
                self.category = category
                self.confidence = confidence
                self.reasoning = reasoning

        return Result(
            category=expense.get("category"),
            confidence=expense.get("confidence"),
            reasoning=expense.get("reasoning"),
        )
