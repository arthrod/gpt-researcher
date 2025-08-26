import json
from typing import Dict, Any
from pydantic import BaseModel

class UserSchema(BaseModel):
    id: int
    name: str
    email: str
    age: int
    is_active: bool

def generate_structured_json(schema: BaseModel, data: Dict[str, Any]) -> str:
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
from collections.abc import Mapping
 from pydantic import BaseModel

def generate_structured_json(schema: type[BaseModel] | BaseModel, data: Mapping[str, Any]) -> str:
     """
     Generate structured JSON output based on provided schema
     """
     try:
-        structured_data = schema(**data)
-        # Convert to JSON string
        # Build a model instance from dict for both v1 and v2
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # class passed
            if hasattr(schema, "model_validate"):
                structured_data = schema.model_validate(dict(data))  # pydantic v2
            else:
                structured_data = schema(**data)  # pydantic v1
        else:
            # instance passed
            structured_data = schema.__class__(**data)

        # Serialize (v2 preferred, fallback for v1)
        if hasattr(structured_data, "model_dump_json"):
            return structured_data.model_dump_json(indent=2)  # pydantic v2
        if hasattr(structured_data, "json"):
            return structured_data.json(indent=2)  # pydantic v1
        # Fallback
        return json.dumps(
            getattr(structured_data, "dict", lambda: dict(structured_data))(),
            indent=2,
        )
     except Exception as e:
        return json.dumps(
            {"error": "Error generating JSON", "detail": f"{e!s}"},
            indent=2,
        )

# Example usage
if __name__ == "__main__":
    sample_data = {
        "id": 1,
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        "is_active": True
    }

    json_output = generate_structured_json(UserSchema, sample_data)
    print("Structured JSON Output:")
    print(json_output)
