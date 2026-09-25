from enum import Enum

from pydantic import BaseModel, Field, field_validator


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

    @classmethod
    @field_validator("category", mode="before")
    def normalize_category(cls, value: str) -> str:
        """Convierte sinónimos comunes del LLM a los valores permitidos del Enum."""
        if isinstance(value, str):
            val_lower = value.lower()

            # Mapeamos variantes que suelen usar los LLMs
            if val_lower in [
                "groceries",
                "alimentación",
                "comida",
                "supermarket",
                "food",
            ]:
                return "food"
            if val_lower in [
                "gasoline",
                "transport",
                "travel",
                "uber",
                "taxi",
                "transportation",
            ]:
                return "transportation"
            if val_lower in ["entertainment", "movies", "games", "concerts"]:
                return "entertainment"
            if val_lower in ["utilities", "electricity", "water", "internet"]:
                return "utilities"
            return val_lower
        return value
