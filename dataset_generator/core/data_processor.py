import json
import math
import random
import uuid
from typing import Dict, List, Any, Optional, Tuple

from tqdm import tqdm

from dataset_generator.utils.constants import (
    logger, DATASET_TEMPLATES,
    SKLEARN_AVAILABLE, FAKER_AVAILABLE, NLTK_AVAILABLE
)
from dataset_generator.utils.helpers import get_nested_value

if SKLEARN_AVAILABLE:
    from sklearn.model_selection import train_test_split

if FAKER_AVAILABLE:
    from faker import Faker

if NLTK_AVAILABLE:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import wordnet

def _map_item_to_template(item: Dict[str, Any], template_structure: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """Map a single input item to the template structure."""
    result = {}

    def _map_structure(structure, result_dict, prefix=""):
        for key, value in structure.items():
            full_key = f"{prefix}{key}"
            if isinstance(value, dict):
                result_dict[key] = {}
                _map_structure(value, result_dict[key], f"{full_key}.")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                result_dict[key] = []
                for i, item_template in enumerate(value):
                    result_dict[key].append({})
                    _map_structure(item_template, result_dict[key][i], f"{full_key}[{i}].")
            else:
                if full_key in mapping:
                    input_field = mapping[full_key]
                    result_dict[key] = get_nested_value(item, input_field, "")
                else:
                    result_dict[key] = ""

    _map_structure(template_structure, result)
    return result

def process_data(input_data: List[Dict[str, Any]], template_name: str, mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    """Process input data according to the template and mapping."""
    if not template_name or not mapping:
        raise ValueError("Template and field mapping must be set before processing")

    logger.info("Processing data...")
    template_structure = DATASET_TEMPLATES[template_name]["structure"]
    output_data = []

    if not input_data:
        return []

    for item in tqdm(input_data, desc="Processing records"):
        output_item = _map_item_to_template(item, template_structure, mapping)
        output_data.append(output_item)

    logger.info(f"Successfully processed {len(output_data)} records")
    return output_data

def split_dataset(data: List[Dict[str, Any]], train_ratio: float = 0.8, valid_ratio: float = 0.1, test_ratio: float = 0.1, random_seed: Optional[int] = None) -> Tuple[List, List, List]:
    """Split dataset into train, validation, and test sets."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn library not available for dataset splitting.")

    if not data:
        raise ValueError("No data to split.")

    total = train_ratio + valid_ratio + test_ratio
    if not math.isclose(total, 1.0):
        logger.warning(f"Ratios sum to {total}, they will be normalized.")
        train_ratio /= total
        valid_ratio /= total
        test_ratio /= total

    train_data, temp_data = train_test_split(
        data, test_size=(1.0 - train_ratio), random_state=random_seed
    )

    if valid_ratio > 0 and test_ratio > 0:
        # Calculate the split proportion for the remaining data
        remaining_ratio = valid_ratio / (valid_ratio + test_ratio)
        valid_data, test_data = train_test_split(
            temp_data, test_size=(1.0 - remaining_ratio), random_state=random_seed
        )
    elif valid_ratio > 0:
        valid_data = temp_data
        test_data = []
    else: # test_ratio > 0 or both are 0
        valid_data = []
        test_data = temp_data

    logger.info(f"Split dataset: {len(train_data)} train, {len(valid_data)} validation, {len(test_data)} test")
    return train_data, valid_data, test_data

def _get_synonyms(word: str) -> List[str]:
    """Helper to get synonyms using NLTK WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            syn_name = lemma.name().replace('_', ' ')
            if syn_name != word:
                synonyms.add(syn_name)
    return list(synonyms)

def _replace_with_synonyms(text: str) -> str:
    """Replace some words in text with synonyms."""
    words = word_tokenize(text)
    new_words = words[:]
    for i, word in enumerate(words):
        if random.random() < 0.2:  # 20% chance to replace
            synonyms = _get_synonyms(word.lower())
            if synonyms:
                new_words[i] = random.choice(synonyms)
    return " ".join(new_words)

def _rephrase_text(text: str, faker_instance) -> str:
    """Rephrase text using Faker."""
    if len(text.split()) > 10:
        return faker_instance.paragraph(nb_sentences=random.randint(1, 3))
    else:
        return faker_instance.sentence(nb_words=len(text.split()))

def _augment_item(item: Dict, method: str, template_name: Optional[str], faker_instance) -> Dict:
    """Augments a single item."""
    new_item = json.loads(json.dumps(item)) # Deep copy
    template_structure = DATASET_TEMPLATES.get(template_name or '', {}).get("structure", {})

    def _process_text_fields(data_dict: Dict, structure_dict: Dict):
        for key, value in data_dict.items():
            if isinstance(value, str) and len(value.split()) > 3:
                # If a template is provided, only augment fields present in the template
                if template_name and key not in structure_dict:
                    continue

                if method == "shuffle":
                    words = value.split()
                    random.shuffle(words)
                    data_dict[key] = " ".join(words)
                elif method == "synonym" and NLTK_AVAILABLE:
                    data_dict[key] = _replace_with_synonyms(value)
                elif method == "rephrase" and FAKER_AVAILABLE:
                    data_dict[key] = _rephrase_text(value, faker_instance)

    _process_text_fields(new_item, template_structure)
    return new_item

def augment_data(data: List[Dict[str, Any]], augmentation_factor: int = 2, methods: Optional[List[str]] = None, template_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Augment dataset using various techniques."""
    if not data:
        raise ValueError("No data to augment.")
    if augmentation_factor <= 1:
        return data

    if not methods:
        methods = ["shuffle"]
        if NLTK_AVAILABLE: methods.append("synonym")
        if FAKER_AVAILABLE: methods.append("rephrase")

    faker_instance = Faker() if "rephrase" in methods and FAKER_AVAILABLE else None

    augmented_data = []
    logger.info(f"Augmenting dataset (factor: {augmentation_factor})...")
    for item in tqdm(data, desc="Augmenting records"):
        for _ in range(augmentation_factor - 1):
            method = random.choice(methods)
            new_item = _augment_item(item, method, template_name, faker_instance)
            augmented_data.append(new_item)

    logger.info(f"Added {len(augmented_data)} augmented records.")
    return data + augmented_data

def anonymize_data(data: List[Dict[str, Any]], fields_to_anonymize: Dict[str, str]) -> List[Dict[str, Any]]:
    """Anonymize sensitive fields in the dataset."""
    if not FAKER_AVAILABLE:
        raise ImportError("Faker library not available for anonymization.")
    if not data:
        raise ValueError("No data to anonymize.")
    if not fields_to_anonymize:
        logger.warning("No fields to anonymize specified.")
        return data

    faker_instance = Faker()
    logger.info("Anonymizing sensitive data...")
    anonymized_count = 0

    for item in tqdm(data, desc="Anonymizing records"):
        for field, field_type in fields_to_anonymize.items():
            if field in item and item[field] is not None:
                try:
                    # Use getattr to call the correct Faker method
                    faker_method = getattr(faker_instance, field_type)
                    item[field] = faker_method()
                    anonymized_count += 1
                except AttributeError:
                    logger.warning(f"Faker does not have a provider for '{field_type}'. Skipping field '{field}'.")
                except Exception as e:
                    logger.error(f"Error anonymizing field '{field}' with type '{field_type}': {e}")

    logger.info(f"Anonymized {anonymized_count} values across {len(fields_to_anonymize)} fields.")
    return data
