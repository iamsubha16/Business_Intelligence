from typing import Dict, Any, List
from . import data_loader
from .llm_handler import LLMHandler

class MetadataExtractionPipeline:
    """
    A pipeline to extract relevant metadata (schemas, tables, columns)
    based on a user's natural language question.
    """

    def __init__(self, metadata_filepath: str):
        """
        Initializes the pipeline with the path to the metadata source.

        Args:
            metadata_filepath: The file path for the JSON metadata.
        """
        print("--- Initializing Metadata Extraction Pipeline ---")
        self.metadata = data_loader.load_metadata_from_json(metadata_filepath)
        self.llm_handler = LLMHandler()
        print("--- Pipeline Initialized Successfully ---")

    def _get_relevant_schemas(self, user_question: str) -> List[str]:
        """
        Step 1: Get relevant schemas from the LLM.
        """
        print("\n>>> STEP 1: Identifying Relevant Schemas <<<")
        schema_descriptions = data_loader.get_schema_descriptions(self.metadata)
        if not schema_descriptions:
            print("WARNING: No schema descriptions found in the metadata.")
            return []
        relevant_schemas = self.llm_handler.select_relevant_schemas(
            user_question, schema_descriptions
        )
        print(f"✅ Relevant schemas identified: {relevant_schemas}")
        return relevant_schemas

    def _get_relevant_tables(self, user_question: str, schema_name: str) -> List[str]:
        """
        Step 2: Get relevant tables for a given schema from the LLM.
        """
        print(f"\n>>> STEP 2: Identifying Relevant Tables for Schema '{schema_name}' <<<")
        table_descriptions = data_loader.get_tables_for_schema(self.metadata, schema_name)
        if not table_descriptions:
            print(f"WARNING: No tables found for schema '{schema_name}'.")
            return []
        relevant_tables = self.llm_handler.select_relevant_tables(
            user_question, table_descriptions
        )
        print(f"✅ Relevant tables identified in '{schema_name}': {relevant_tables}")
        return relevant_tables

    def _extract_column_metadata(self, schema_name: str, table_name: str) -> Dict[str, str]:
        """
        Step 3: Extract column names and descriptions for a given table.
        """
        print(f"\n>>> STEP 3: Extracting Column Metadata for Table '{schema_name}.{table_name}' <<<")
        column_details = data_loader.get_columns_for_table(self.metadata, schema_name, table_name)
        if not column_details:
             print(f"WARNING: No columns found for table '{schema_name}.{table_name}'.")
        else:
            print(f"✅ Extracted {len(column_details)} columns for '{schema_name}.{table_name}'.")
        return column_details

    def run(self, user_question: str) -> Dict[str, Any]:
        """
        Executes the entire metadata extraction pipeline.

        Args:
            user_question: The natural language question from the user.

        Returns:
            A dictionary containing the extracted metadata, structured by
            schema and table.
        """
        print(f"\n--- Running Pipeline for User Question: '{user_question}' ---")
        final_metadata = {}

        # Step 1: Identify relevant schemas
        relevant_schemas = self._get_relevant_schemas(user_question)

        if not relevant_schemas:
            print("\n--- Pipeline Finished: No relevant schemas could be determined. ---")
            return {}

        # Step 2 & 3: For each relevant schema, find tables and extract columns
        for schema_name in relevant_schemas:
            final_metadata[schema_name] = {}
            relevant_tables = self._get_relevant_tables(user_question, schema_name)
            
            for table_name in relevant_tables:
                column_metadata = self._extract_column_metadata(schema_name, table_name)
                final_metadata[schema_name][table_name] = column_metadata
        
        print("\n--- Pipeline Execution Finished ---")
        return final_metadata

def main():
    """
    Main function to execute the pipeline with an example question.
    """
    metadata_file = "sample_data.json"
    user_question = "What is the growth percentage in sales and customers in each quarter in 2024?"

    pipeline = MetadataExtractionPipeline(metadata_filepath=metadata_file)
    extracted_metadata = pipeline.run(user_question=user_question)

    print("\n\n==============================================")
    print("      Final Extracted Metadata")
    print("==============================================")
    import json
    print(json.dumps(extracted_metadata, indent=2))
    print("==============================================")


# if __name__ == "__main__":
#     main()
