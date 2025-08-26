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
<<<<<<< HEAD
    Generate structured JSON output based on provided schema

    Args:
        schema: Pydantic model defining the schema structure
        data: Dictionary containing the data to be structured

=======
    Generate a JSON string from a Pydantic schema class and input data.
    
    Attempts to instantiate the given Pydantic model class with `data` and returns a pretty-printed JSON string of the resulting model's dict. On failure returns a string beginning with "Error generating JSON: " followed by the exception message.
    
    Parameters:
        schema: A Pydantic model class (subclass of BaseModel), not an instance.
        data: Mapping of field names to values to pass to the model constructor.
    
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
    Returns:
        A JSON-formatted string of the structured data, or an error string if instantiation/serialization fails.
    """
    try:
        # Create instance of schema with provided data
        structured_data = schema(**data)
        # Convert to JSON string
        return json.dumps(structured_data.dict(), indent=2)
    except Exception as e:
        return f"Error generating JSON: {e!s}"
<<<<<<< HEAD

=======
>>>>>>> 1027e1d0 (Fix linting issues)

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
