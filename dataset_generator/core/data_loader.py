import os
import json
import csv
import sqlite3
from typing import List, Dict, Any, Optional

import pandas as pd

from dataset_generator.utils.constants import (
    logger, INPUT_DIR, INPUT_FORMATS,
    PARQUET_AVAILABLE, HDF5_AVAILABLE, XML_AVAILABLE, XML_AVAILABLE_BASIC,
    YAML_AVAILABLE, AVRO_AVAILABLE
)

if PARQUET_AVAILABLE:
    import pyarrow.parquet as pq
if HDF5_AVAILABLE:
    import h5py
if XML_AVAILABLE:
    from lxml import etree
elif XML_AVAILABLE_BASIC:
    import xml.etree.ElementTree as ET
if YAML_AVAILABLE:
    import yaml
if AVRO_AVAILABLE:
    import fastavro

def _xml_to_dict(element):
    """Convert XML element to dictionary."""
    result = {}
    for key, value in element.attrib.items():
        result[f"@{key}"] = value
    if element.text and element.text.strip():
        result["#text"] = element.text.strip()
    for child in element:
        child_data = _xml_to_dict(child)
        if child.tag in result:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data
    return result

def load_data(file_path: str, format_type: Optional[str] = None, table_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads data from a file into a list of dictionaries.
    """
    if not os.path.isabs(file_path):
        file_path = os.path.join(INPUT_DIR, os.path.basename(file_path))

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if not format_type:
        ext = os.path.splitext(file_path)[1].lower()[1:]
        if ext in ['xlsx', 'xls']:
            format_type = 'excel'
        elif ext in INPUT_FORMATS:
            format_type = ext
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    logger.info(f"Loading data from {file_path} as {format_type} format")
    input_data = []

    try:
        if format_type == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if 'data' in data and isinstance(data['data'], list):
                        input_data = data['data']
                    else:
                        input_data = [data]
                else:
                    input_data = data

        elif format_type == 'jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        input_data.append(json.loads(line))

        elif format_type == 'csv':
            df = pd.read_csv(file_path)
            input_data = df.to_dict(orient='records')

        elif format_type == 'tsv':
            df = pd.read_csv(file_path, sep='\t')
            input_data = df.to_dict(orient='records')

        elif format_type == 'excel':
            df = pd.read_excel(file_path)
            input_data = df.to_dict(orient='records')

        elif format_type == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                input_data = [{"text": line.strip()} for line in lines if line.strip()]

        elif format_type == 'parquet':
            if not PARQUET_AVAILABLE:
                raise ImportError("PyArrow library not available for Parquet format.")
            table = pq.read_table(file_path)
            df = table.to_pandas()
            input_data = df.to_dict(orient='records')

        elif format_type == 'hdf5':
            if not HDF5_AVAILABLE:
                raise ImportError("h5py library not available for HDF5 format.")
            with h5py.File(file_path, 'r') as f:
                if not table_name:
                    if not f.keys():
                        raise ValueError("HDF5 file contains no objects.")
                    dataset_name = list(f.keys())[0]
                else:
                    dataset_name = table_name

                if dataset_name not in f:
                    raise ValueError(f"Object '{dataset_name}' not found in HDF5 file.")

                h5_object = f[dataset_name]
                if isinstance(h5_object, h5py.Dataset):
                    if hasattr(h5_object, 'dtype') and h5_object.dtype.names:
                        input_data = [dict(row) for row in h5_object]
                    else:
                        input_data = [{"value": v.item() if hasattr(v, 'item') else v} for v in h5_object[:]]
                else:
                    raise ValueError(f"The specified HDF5 object '{dataset_name}' is a Group, not a Dataset.")


        elif format_type == 'sqlite':
            conn = sqlite3.connect(file_path)
            conn.row_factory = sqlite3.Row
            if not table_name:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                if not tables:
                    raise ValueError("No tables found in SQLite database")
                table_name = tables[0][0]
            # Validate table_name against the list of tables in the database
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            valid_tables = {row[0] for row in cursor.fetchall()}
            if table_name not in valid_tables:
                raise ValueError(f"Table '{table_name}' not found in SQLite database")
            cursor = conn.execute(f"SELECT * FROM \"{table_name}\"")
            input_data = [dict(row) for row in cursor.fetchall()]
            conn.close()

        elif format_type == 'xml':
            if not XML_AVAILABLE and not XML_AVAILABLE_BASIC:
                raise ImportError("An XML parsing library is not available.")

            if XML_AVAILABLE:
                tree = etree.parse(file_path)
                root = tree.getroot()
            else:
                tree = ET.parse(file_path)
                root = tree.getroot()

            for element in root:
                input_data.append(_xml_to_dict(element))

        elif format_type == 'yaml':
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML library not available for YAML format.")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if isinstance(data, list):
                    input_data = data
                elif isinstance(data, dict):
                    if 'data' in data and isinstance(data['data'], list):
                        input_data = data['data']
                    else:
                        input_data = [data]

        elif format_type == 'avro':
            if not AVRO_AVAILABLE:
                raise ImportError("fastavro library not available for Avro format.")
            with open(file_path, 'rb') as f:
                input_data = [record for record in fastavro.reader(f)]

        else:
            raise ValueError(f"Unsupported input format: {format_type}")

        logger.info(f"Successfully loaded {len(input_data)} records")
        return input_data

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
