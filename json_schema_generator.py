import json

from typing import Any

from pydantic import BaseModel


class UserSchema(BaseModel):
    id: int
    name: str
    email: str
    age: int
    is_active: bool


def generate_structured_json(schema: BaseModel, data: dict[str, Any]) -> str:
    """
    Generate structured JSON output based on provided schema

    Args:
        schema: Pydantic model defining the schema structure
        data: Dictionary containing the data to be structured

    Returns:
        str: JSON string with structured data
    """
    try:
        # Create instance of schema with provided data
        structured_data = schema(**data)
        # Convert to JSON string
        return json.dumps(structured_data.dict(), indent=2)
    except Exception as e:
import json
from typing import Any
from pydantic import BaseModel

def generate_structured_json(schema: type[BaseModel], data: dict[str, Any]) -> str:
    try:
        structured_data = schema(**data)
        # Pydantic v2: model_dump; fallback to dict() if v1
        to_dict = getattr(structured_data, "model_dump", None)
        payload = to_dict() if callable(to_dict) else structured_data.dict()
        return json.dumps(payload, indent=2)
    except Exception as e:
        raise ValueError(f"Error generating JSON: {e!s}") from e


# Example usage
if __name__ == "__main__":
    sample_data = {
        "id": 1,
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        "is_active": True,
    }

    json_output = generate_structured_json(UserSchema, sample_data)
    print("Structured JSON Output:")
    print(json_output)
