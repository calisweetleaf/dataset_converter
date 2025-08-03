from typing import Dict, Any

def get_nested_value(data: Dict, path: str, default=None):
    """Retrieves a nested value from a dictionary using a dot-notation path."""
    parts = path.split('.')
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            try:
                current = current[int(part)]
            except IndexError:
                return default # Index out of bounds for list
        else:
            return default
    return current

def set_nested_value(data: Dict, path: str, value: Any):
    """Sets a nested value in a dictionary using a dot-notation path.
    Creates intermediate dictionaries/lists if they don't exist.
    """
    parts = path.split('.')
    current = data
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            current[part] = value
        else:
            if part not in current or not isinstance(current[part], (dict, list)):
                # Try to infer if the next part is an array index
                if i + 1 < len(parts) and parts[i+1].isdigit():
                    current[part] = []
                else:
                    current[part] = {}
            current = current[part]
