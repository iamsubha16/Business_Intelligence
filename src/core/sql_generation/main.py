# import json

# # Custom modules for each pipeline stage
# from ..metadata_extraction.pipeline import MetadataExtractionPipeline
# from .sql_generator import SQLGenerator
# from ..exceptions import BusinessIntvException
# from ..logging_config import setup_logging, get_logger

# # --- Setup Logging ---
# # This should be the first thing to run to ensure all subsequent
# # modules have logging configured.
# setup_logging()
# logger = get_logger(__name__)


# def run_full_pipeline():
#     """
#     Executes the complete end-to-end pipeline:
#     1. Extracts relevant metadata based on a user question.
#     2. Generates an SQL query based on the extracted metadata.
#     """
#     USER_QUESTION = "What is the growth percentage in sales and customers in each quarter in 2024?"
#     METADATA_FILE = "sample_data.json"
    
#     logger.info("--- STARTING END-TO-END PIPELINE ---")
    
#     try:
#         # --- Stage 1: Metadata Extraction ---
#         logger.info("--- Running Stage 1: Metadata Extraction ---")
#         metadata_pipeline = MetadataExtractionPipeline(METADATA_FILE)
#         extracted_metadata = metadata_pipeline.run(USER_QUESTION)
        
#         if not extracted_metadata:
#             logger.warning("Stage 1 (Metadata Extraction) did not return any metadata. Halting pipeline.")
#             return

#         logger.info("--- Stage 1 Complete. Extracted Metadata: ---")
#         logger.info(f"\n{json.dumps(extracted_metadata, indent=4)}\n")

#         # --- Stage 2: SQL Generation ---
#         logger.info("--- Running Stage 2: SQL Generation ---")
#         sql_pipeline = SQLGenerator()
#         generated_sql = sql_pipeline.generate_query(USER_QUESTION, extracted_metadata)

#         logger.info("--- Stage 2 Complete. Final Generated SQL Query: ---")
#         logger.info(f"\n---\n{generated_sql}\n---")

#     except BusinessIntvException as e:
#         # Catches any custom application errors (e.g., API key, data loading, LLM response)
#         logger.error(f"A critical application error occurred during the pipeline: {e}", exc_info=False)
#     except Exception as e:
#         # Catches any unexpected errors
#         logger.critical(f"An unexpected and unhandled error occurred: {e}", exc_info=True)
#     finally:
#         logger.info("--- PIPELINE RUN FINISHED ---")


# if __name__ == "__main__":
#     run_full_pipeline()


import json

# Custom modules for each pipeline stage
from ..metadata_extraction.pipeline import MetadataExtractionPipeline
from .sql_generator import SQLGenerator
from ..database.db_handler import DatabaseHandler
from ..exceptions import BusinessIntvException
from ..logging_config import setup_logging, get_logger

# --- Setup Logging ---
setup_logging()
logger = get_logger(__name__)


def run_full_pipeline():
    """
    Executes the complete end-to-end pipeline:
    1. Extracts relevant metadata based on a user question.
    2. Generates an SQL query based on the extracted metadata.
    3. Executes the generated SQL query on the database.
    """
    USER_QUESTION = "List each customer's CustomerName and total orders in 2022 where OrderStatus is NOT 'Cancelled'. Show only those with more than 2 such orders."
    METADATA_FILE = "sample_data.json"
    
    logger.info("--- STARTING END-TO-END PIPELINE ---")
    
    try:
        # --- Stage 1: Metadata Extraction ---
        logger.info("--- Running Stage 1: Metadata Extraction ---")
        metadata_pipeline = MetadataExtractionPipeline(METADATA_FILE)
        extracted_metadata = metadata_pipeline.run(USER_QUESTION)
        
        if not extracted_metadata:
            logger.warning("Stage 1 (Metadata Extraction) did not return any metadata. Halting pipeline.")
            return

        logger.info("--- Stage 1 Complete. Extracted Metadata: ---")
        logger.info(f"\n{json.dumps(extracted_metadata, indent=4)}\n")

        # --- Stage 2: SQL Generation ---
        logger.info("--- Running Stage 2: SQL Generation ---")
        sql_pipeline = SQLGenerator()
        generated_sql = sql_pipeline.generate_query(USER_QUESTION, extracted_metadata)

        if not generated_sql or not generated_sql.strip():
            logger.warning("Stage 2 (SQL Generation) did not produce a query. Halting pipeline.")
            return

        logger.info("--- Stage 2 Complete. Final Generated SQL Query: ---")
        logger.info(f"\n---\n{generated_sql}\n---")

        # --- Stage 3: SQL Execution ---
        logger.info("--- Running Stage 3: SQL Execution ---")
        db_handler = DatabaseHandler()
        query_results = db_handler.execute_query(generated_sql)

        logger.info("--- Stage 3 Complete. Query Execution Results: ---")
        # Use default=str to handle non-serializable types like dates or decimals
        logger.info(f"\n{json.dumps(query_results, indent=4, default=str)}\n")

    except BusinessIntvException as e:
        # Catches any custom application errors
        logger.error(f"A critical application error occurred during the pipeline: {e}", exc_info=False)
    except Exception as e:
        # Catches any unexpected errors
        logger.critical(f"An unexpected and unhandled error occurred: {e}", exc_info=True)
    finally:
        logger.info("--- PIPELINE RUN FINISHED ---")


if __name__ == "__main__":
    run_full_pipeline()

