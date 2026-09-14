from enum import Enum

from pydantic import BaseModel, Field


class CategoryEnum(str, Enum):
    FOOD = "food"
    TRANSPORTATION = "transportation"
    ENTERTAINMENT = "entertainment"
    UTILITIES = "utilities"
    OTHER = "other"


class ExpenseClassification(BaseModel):
    category: CategoryEnum
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
