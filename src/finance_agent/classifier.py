from enum import Enum

from pydantic import BaseModel


class CategoryEnum(str, Enum):
    FOOD = "food"
    TRANSPORTATION = "transportation"
    ENTERTAINMENT = "entertainment"
    UTILITIES = "utilities"
    OTHER = "other"


class ExpenseClassification(BaseModel):
    category: CategoryEnum
    confidence: float
    reasoning: str
