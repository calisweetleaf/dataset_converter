import os
import logging

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(BASE_DIR, 'dataset_input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'dataset_output')
LOG_DIR = os.path.join(BASE_DIR, 'dataset_logs')
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, 'dataset_generator.log'))
    ]
)
logger = logging.getLogger("DatasetGenerator")

# --- Availability Flags ---
try:
    import pyarrow
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from lxml import etree
    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False
    try:
        import xml.etree.ElementTree as ET
        XML_AVAILABLE_BASIC = True
    except ImportError:
        XML_AVAILABLE_BASIC = False

try:
    import fastavro
    AVRO_AVAILABLE = True
except ImportError:
    AVRO_AVAILABLE = False

try:
    import matplotlib
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


# --- Formats ---
OUTPUT_FORMATS = {
    "jsonl": "JSON Lines format (each line is a valid JSON object)",
    "json": "Single JSON file with array of examples",
    "csv": "CSV format with headers",
    "txt": "Plain text format",
    "parquet": "Apache Parquet columnar storage format",
    "hdf5": "Hierarchical Data Format version 5",
    "sqlite": "SQLite database file",
    "xml": "XML format",
    "yaml": "YAML format",
    "avro": "Apache Avro data serialization format",
    "excel": "Microsoft Excel format (.xlsx)",
    "tsv": "Tab-separated values format",
}

INPUT_FORMATS = {
    "json": "JSON file (array or object)",
    "jsonl": "JSON Lines file (each line is a JSON object)",
    "csv": "CSV file with headers",
    "txt": "Plain text file",
    "excel": "Excel file (.xlsx, .xls)",
    "parquet": "Apache Parquet columnar storage format",
    "hdf5": "Hierarchical Data Format version 5",
    "sqlite": "SQLite database file (requires table name)",
    "xml": "XML format",
    "yaml": "YAML format",
    "avro": "Apache Avro data serialization format",
    "tsv": "Tab-separated values format",
}

# --- Templates ---
DATASET_TEMPLATES = {
    "instruction": {
        "description": "Simple instruction-following format (prompt, completion)",
        "structure": {"prompt": "", "completion": ""}
    },
    "chat": {
        "description": "Chat conversation format (messages: user, assistant)",
        "structure": {"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
    },
    "qa": {
        "description": "Question-Answer format (question, answer)",
        "structure": {"question": "", "answer": ""}
    },
    "openai_fine_tuning": {
        "description": "OpenAI fine-tuning format (system, user, assistant messages)",
        "structure": {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
    },
    "anthropic_format": {
        "description": "Anthropic Claude format (input, output)",
        "structure": {"input": "", "output": ""}
    },
    "alpaca": {
        "description": "Alpaca instruction format (instruction, input, output)",
        "structure": {"instruction": "", "input": "", "output": ""}
    },
    "llama": {
        "description": "Llama instruction format (single 'text' field)",
        "structure": {"text": ""}
    },
    "mistral": {
        "description": "Mistral instruction format (instruction, response)",
        "structure": {"instruction": "", "response": ""}
    },
    "custom": {
        "description": "Custom format defined by user",
        "structure": {}
    }
}
