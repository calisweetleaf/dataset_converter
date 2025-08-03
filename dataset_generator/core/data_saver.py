import os
import json
import sqlite3
from typing import List, Dict, Any, Optional

import pandas as pd

from dataset_generator.utils.constants import (
    logger, OUTPUT_DIR,
    PARQUET_AVAILABLE, HDF5_AVAILABLE, XML_AVAILABLE,
    YAML_AVAILABLE, AVRO_AVAILABLE
)

if PARQUET_AVAILABLE:
    import pyarrow as pa
    import pyarrow.parquet as pq
if HDF5_AVAILABLE:
    import h5py
if XML_AVAILABLE:
    from lxml import etree
if YAML_AVAILABLE:
    import yaml
if AVRO_AVAILABLE:
    import fastavro

def save_data(data: List[Dict[str, Any]], output_path: str, format_type: str) -> None:
    """
    Saves data to a file in the specified format.
    """
    if not os.path.isabs(output_path):
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(output_path))

    if not data:
        raise ValueError("No data to save.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Saving data to {output_path} as {format_type} format")

    try:
        if format_type == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        elif format_type == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        elif format_type in ['csv', 'tsv', 'excel']:
            df = pd.DataFrame(data)
            if format_type == 'csv':
                df.to_csv(output_path, index=False)
            elif format_type == 'tsv':
                df.to_csv(output_path, sep='\t', index=False)
            elif format_type == 'excel':
                df.to_excel(output_path, index=False)

        elif format_type == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        elif format_type == 'parquet':
            if not PARQUET_AVAILABLE:
                raise ImportError("PyArrow library not available for Parquet format.")
            df = pd.DataFrame(data)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, output_path)

        elif format_type == 'hdf5':
            if not HDF5_AVAILABLE:
                raise ImportError("h5py library not available for HDF5 format.")
            df = pd.DataFrame(data)
            df.to_hdf(output_path, key='data', mode='w')

        elif format_type == 'sqlite':
            df = pd.DataFrame(data)
            conn = sqlite3.connect(output_path)
            df.to_sql('data', conn, if_exists='replace', index=False)
            conn.close()

        elif format_type == 'xml':
            if not XML_AVAILABLE:
                raise ImportError("lxml library not available for XML format.")

            def _add_xml_element(parent, data_item):
                if isinstance(data_item, dict):
                    for k, v in data_item.items():
                        child = etree.SubElement(parent, k)
                        _add_xml_element(child, v)
                elif isinstance(data_item, list):
                    for sub_item in data_item:
                        child = etree.SubElement(parent, "item")
                        _add_xml_element(child, sub_item)
                else:
                    parent.text = str(data_item)

            root = etree.Element("dataset")
            for item in data:
                record = etree.SubElement(root, "record")
                _add_xml_element(record, item)

            tree = etree.ElementTree(root)
            tree.write(output_path, pretty_print=True, encoding='utf-8', xml_declaration=True)

        elif format_type == 'yaml':
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML library not available for YAML format.")
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False)

        elif format_type == 'avro':
            if not AVRO_AVAILABLE:
                raise ImportError("fastavro library not available for Avro format.")
            if not data:
                return

            # A more robust schema inference would be needed for production
            # For now, let's assume a simple schema from the first record
            first_record = data[0]
            schema_fields = []
            for key, value in first_record.items():
                field_type = "string"
                if isinstance(value, int):
                    field_type = "long"
                elif isinstance(value, float):
                    field_type = "double"
                elif isinstance(value, bool):
                    field_type = "boolean"
                schema_fields.append({"name": key, "type": ["null", field_type]})

            schema = {
                "type": "record",
                "name": "Dataset",
                "fields": schema_fields,
            }

            with open(output_path, 'wb') as f:
                fastavro.writer(f, schema, data)

        else:
            raise ValueError(f"Unsupported output format: {format_type}")

        logger.info(f"Successfully saved {len(data)} records to {output_path}")

    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise
