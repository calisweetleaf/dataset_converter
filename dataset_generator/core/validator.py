from typing import Dict, List, Any, Optional
from collections import Counter

from dataset_generator.utils.constants import logger, DATASET_TEMPLATES

def calculate_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics about the dataset."""
    if not data:
        return {}

    stats = {
        "record_count": len(data),
        "field_counts": Counter(),
        "field_types": {},
        "missing_values": Counter(),
    }

    all_field_names = set()
    for item in data:
        all_field_names.update(item.keys())
        stats['field_counts'].update(item.keys())

    for field in all_field_names:
        non_null_values = [item.get(field) for item in data if item.get(field) is not None]

        stats['missing_values'][field] = len(data) - len(non_null_values)

        if non_null_values:
            # Infer type from the first non-null value
            sample_value = non_null_values[0]
            if isinstance(sample_value, str):
                stats["field_types"][field] = "string"
            elif isinstance(sample_value, (int, float)):
                stats["field_types"][field] = "number"
            elif isinstance(sample_value, bool):
                stats["field_types"][field] = "boolean"
            elif isinstance(sample_value, list):
                stats["field_types"][field] = "array"
            elif isinstance(sample_value, dict):
                stats["field_types"][field] = "object"
            else:
                stats["field_types"][field] = "unknown"

    logger.info(f"Calculated statistics for {stats['record_count']} records.")
    return stats

def validate_dataset(data: List[Dict[str, Any]], custom_rules: Optional[List[Dict]] = None) -> List[str]:
    """
    Validates the dataset against a set of rules.
    Returns a list of validation error messages.
    """
    if not data:
        return ["Dataset is empty. Nothing to validate."]

    errors = []
    # Basic validation (can be expanded)
    # 1. Check for duplicate records (exact match)
    seen = set()
    for i, item in enumerate(data):
        # Convert dict to a frozenset of items to make it hashable
        item_tuple = frozenset(item.items())
        if item_tuple in seen:
            errors.append(f"Row {i+1}: Duplicate record found.")
        else:
            seen.add(item_tuple)

    # 2. Check for inconsistent field presence
    if data:
        first_item_keys = set(data[0].keys())
        for i, item in enumerate(data[1:]):
            if set(item.keys()) != first_item_keys:
                errors.append(f"Row {i+2}: Inconsistent set of fields compared to the first row.")

    # TODO: Implement custom rule validation logic here
    if custom_rules:
        logger.warning("Custom rule validation is not yet fully implemented.")

    logger.info(f"Validation complete. Found {len(errors)} potential issues.")
    return errors

def fix_dataset(data: List[Dict[str, Any]], template_name: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Attempts to automatically fix common dataset issues.
    Returns the fixed data and a list of actions taken.
    """
    if not data:
        return [], ["Dataset is empty. Nothing to fix."]

    fixed_data = [item.copy() for item in data] # Work on a copy
    actions = []

    template_structure = DATASET_TEMPLATES.get(template_name or '', {}).get("structure", {})

    for i, item in enumerate(fixed_data):
        # 1. Fix None values by converting them to empty strings
        for key, value in item.items():
            if value is None:
                item[key] = ""
                actions.append(f"Row {i+1}, Field '{key}': Converted None to empty string.")

        # 2. If a template is provided, ensure all required fields exist
        if template_structure:
            for key, default_value in template_structure.items():
                if key not in item:
                    item[key] = default_value
                    actions.append(f"Row {i+1}: Added missing template field '{key}'.")

    logger.info(f"Auto-fixing complete. Performed {len(actions)} actions.")
    return fixed_data, actions
