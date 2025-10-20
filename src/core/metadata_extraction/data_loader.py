import json
from typing import Dict, Any

def load_metadata_from_json(filepath: str) -> Dict[str, Any]:
    """
    Loads metadata from a specified JSON file.

    Args:
        filepath: The path to the JSON file.

    Returns:
        A dictionary containing the loaded metadata.
        
    Raises:
        FileNotFoundError: If the file is not found.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: The file at {filepath} is not a valid JSON file.")
        raise

def get_schema_descriptions(metadata: Dict[str, Any]) -> Dict[str, str]:
    """
    Extracts schema names and their descriptions from the metadata.

    Args:
        metadata: The complete metadata dictionary.

    Returns:
        A dictionary mapping schema names to their descriptions.
    """
    return {schema['name']: schema['description'] for schema in metadata.get('schemas', [])}

def get_tables_for_schema(metadata: Dict[str, Any], schema_name: str) -> Dict[str, str]:
    """
    Retrieves all tables and their descriptions for a given schema.

    Args:
        metadata: The complete metadata dictionary.
        schema_name: The name of the schema to get tables from.

    Returns:
        A dictionary mapping table names to their descriptions for the specified schema.
    """
    for schema in metadata.get('schemas', []):
        if schema['name'] == schema_name:
            return {table['name']: table['description'] for table in schema.get('tables', [])}
    return {}

def get_columns_for_table(metadata: Dict[str, Any], schema_name: str, table_name: str) -> Dict[str, str]:
    """
    Retrieves all columns and their descriptions for a given table in a schema.

    Args:
        metadata: The complete metadata dictionary.
        schema_name: The name of the schema.
        table_name: The name of the table.

    Returns:
        A dictionary mapping column names to their descriptions for the specified table.
    """
    for schema in metadata.get('schemas', []):
        if schema['name'] == schema_name:
            for table in schema.get('tables', []):
                if table['name'] == table_name:
                    return {column['name']: column['description'] for column in table.get('columns', [])}
    return {}
