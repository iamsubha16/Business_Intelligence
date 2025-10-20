"""
SQL Generator module for the Metadata Extraction Pipeline.

This module handles the generation of SQL queries from natural language questions
using LLM capabilities and extracted metadata.
"""

import json
import os
import re
from typing import Dict, Any, Optional

# LangChain components for building the pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Custom modules
from ..logging_config import get_logger
from ..exceptions import (
    APIKeyNotFoundError,
    LLMConnectionError,
    LLMResponseError,
    SQLGenerationError
)

# Initialize logger for this module
logger = get_logger(__name__)


class SQLGenerator:
    """
    Handles the generation of SQL queries based on user questions and extracted metadata.
    
    This class uses LangChain and Google's Gemini model to convert natural language
    questions into executable SQL queries, using database metadata as context.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash-preview-09-2025"):
        """
        Initializes the LangChain-based SQL Generator client.
        
        Args:
            model_name: The name of the Gemini model to use.
        
        Raises:
            APIKeyNotFoundError: If the GOOGLE_API_KEY environment variable is not set.
            LLMConnectionError: If there's an error initializing the model.
        """
        logger.info("Initializing SQLGenerator", model_name=model_name)
        
        # Validate API key exists
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error(
                "GOOGLE_API_KEY environment variable not found",
                key_name="GOOGLE_API_KEY"
            )
            raise APIKeyNotFoundError(
                "GOOGLE_API_KEY environment variable not found. "
                "Please set it in your .env file or environment."
            )
        
        self.model_name = model_name
        
        try:
            # Initialize the ChatGoogleGenerativeAI model with temperature=0 for consistency
            self.model = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0,
                convert_system_message_to_human=True
            )
            self.parser = StrOutputParser()
            
            logger.info(
                "SQLGenerator initialized successfully",
                model_name=self.model_name,
                temperature=0
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize SQL Generator model",
                model_name=model_name,
                error=str(e),
                exc_info=True
            )
            raise LLMConnectionError(
                f"Failed to initialize SQL Generator model '{model_name}': {str(e)}"
            ) from e

    def _create_chain(self):
        """
        Creates the LangChain pipeline for SQL generation.
        
        Returns:
            A LangChain pipeline (LCEL chain) for SQL generation.
            
        Raises:
            LLMConnectionError: If pipeline creation fails.
        """
        try:
            template = """
You are a world-class SQL generation expert. Your task is to write a precise and runnable SQL query
that answers the user's question.

You MUST adhere to the following rules:
1. Use ONLY the tables and columns provided in the "METADATA" section. Do not hallucinate any table or column names.
2. The table names and column names are not intuitive, so you MUST rely on their descriptions to understand their meaning.
3. Your response MUST be ONLY the SQL query. Do not include any explanations, markdown formatting (like ```sql), or any other text.
4. Ensure proper SQL syntax with correct JOINs, WHERE clauses, and aggregations as needed.
5. Use fully qualified table names (schema.table) when referencing tables.
6. Include appropriate filtering, grouping, and ordering based on the user's question.
7. The query must be syntactically valid in MySQL. Use MySQL-specific functions and casting (e.g., CAST(... AS SIGNED) instead of CAST(... AS INTEGER)).

---
USER QUESTION:
{user_question}
---
METADATA:
{metadata}
---
SQL QUERY:
"""
            
            prompt = ChatPromptTemplate.from_template(template=template)
            chain = prompt | self.model | self.parser
            
            logger.debug("SQL generation LangChain pipeline created successfully")
            return chain
            
        except Exception as e:
            logger.error(
                "Failed to create SQL generation pipeline",
                error=str(e),
                exc_info=True
            )
            raise LLMConnectionError(
                f"Failed to create SQL generation pipeline: {str(e)}"
            ) from e

    def _validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Validates that the metadata structure is correct and non-empty.
        
        Args:
            metadata: The extracted metadata dictionary.
            
        Returns:
            True if metadata is valid, False otherwise.
        """
        if not metadata:
            logger.warning("Metadata is empty")
            return False
        
        if not isinstance(metadata, dict):
            logger.warning(
                "Metadata has invalid type",
                expected_type="dict",
                actual_type=type(metadata).__name__
            )
            return False
        
        # Check if there's at least one schema with at least one table
        has_content = False
        for schema_name, tables in metadata.items():
            if isinstance(tables, dict) and tables:
                has_content = True
                break
        
        if not has_content:
            logger.warning("Metadata contains no schemas or tables")
            return False
        
        logger.debug(
            "Metadata validation successful",
            num_schemas=len(metadata),
            total_tables=sum(len(tables) for tables in metadata.values() if isinstance(tables, dict))
        )
        return True

    def _clean_sql_response(self, response: str) -> str:
        """
        Cleans the SQL response from the LLM by removing markdown formatting
        and extra whitespace.
        
        Args:
            response: The raw response from the LLM.
            
        Returns:
            Cleaned SQL query string.
        """
        if not response:
            logger.warning("Empty response received from LLM")
            return ""
        
        # Remove markdown code blocks
        cleaned = response.strip()
        cleaned = re.sub(r'^```sql\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^```\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        
        # Remove any leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # Log if cleaning was necessary
        if cleaned != response.strip():
            logger.debug(
                "SQL response was cleaned",
                original_length=len(response),
                cleaned_length=len(cleaned)
            )
        
        return cleaned

    def _validate_sql_query(self, query: str) -> bool:
        """
        Performs basic validation on the generated SQL query.
        
        Args:
            query: The SQL query to validate.
            
        Returns:
            True if query appears valid, False otherwise.
        """
        if not query:
            logger.warning("SQL query is empty")
            return False
        
        # Check for common SQL keywords
        query_upper = query.upper()
        has_select = 'SELECT' in query_upper
        has_from = 'FROM' in query_upper
        
        if not has_select:
            logger.warning("Generated SQL query missing SELECT clause")
            return False
        
        if not has_from:
            logger.warning("Generated SQL query missing FROM clause")
            return False
        
        logger.debug("SQL query validation passed")
        return True

    # Main
    def generate_query(
        self,
        user_question: str,
        extracted_metadata: Dict[str, Any]
    ) -> str:
        """
        Generates an SQL query using the LLM.

        Args:
            user_question: The original question from the user.
            extracted_metadata: The metadata (schemas, tables, columns, descriptions)
                               from the metadata extraction pipeline.

        Returns:
            A string containing the generated SQL query.
            
        Raises:
            SQLGenerationError: If metadata is invalid or SQL generation fails.
            LLMConnectionError: If there's a network or API connection issue.
            LLMResponseError: If the LLM response is invalid or malformed.
        """
        logger.info(
            "Starting SQL query generation",
            user_question=user_question
        )
        
        # Validate inputs
        if not user_question or not user_question.strip():
            logger.error("User question is empty or None")
            raise SQLGenerationError("User question cannot be empty")
        
        # Validate metadata
        if not self._validate_metadata(extracted_metadata):
            logger.error(
                "Invalid or empty metadata provided for SQL generation",
                metadata_keys=list(extracted_metadata.keys()) if extracted_metadata else []
            )
            raise SQLGenerationError(
                "Invalid or empty metadata. Cannot generate SQL query without valid metadata."
            )
        
        try:
            # Create the chain
            chain = self._create_chain()
            
            # Format the metadata for the prompt
            metadata_context = json.dumps(extracted_metadata, indent=4)
            logger.debug(
                "Metadata formatted for prompt",
                metadata_size=len(metadata_context)
            )
            
            logger.info("Invoking LLM chain for SQL generation")
            
            # Invoke the chain
            response = chain.invoke({
                "user_question": user_question,
                "metadata": metadata_context
            })
            
            if not response:
                logger.error("LLM returned empty response")
                raise LLMResponseError("LLM returned an empty response")
            
            logger.debug(
                "Received response from LLM",
                response_length=len(response)
            )
            
            # Clean the response
            clean_query = self._clean_sql_response(response)
            
            # Validate the generated query
            if not self._validate_sql_query(clean_query):
                logger.error(
                    "Generated SQL query failed validation",
                    query=clean_query
                )
                raise SQLGenerationError(
                    "Generated SQL query failed validation checks"
                )
            
            logger.info(
                "SQL query generated successfully",
                query_length=len(clean_query)
            )
            logger.debug("Generated SQL query", query=clean_query)
            
            return clean_query
            
        except SQLGenerationError:
            # Re-raise our custom errors
            raise
            
        except LLMConnectionError:
            # Re-raise connection errors
            logger.error("LLM connection error during SQL generation")
            raise
            
        except Exception as e:
            logger.error(
                "Unexpected error during SQL generation",
                user_question=user_question,
                error=str(e),
                exc_info=True
            )
            raise SQLGenerationError(
                f"Failed to generate SQL query: {str(e)}"
            ) from e

    def generate_query_with_retry(
        self,
        user_question: str,
        extracted_metadata: Dict[str, Any],
        max_retries: int = 2
    ) -> Optional[str]:
        """
        Generates an SQL query with automatic retry logic.
        
        This method attempts to generate a SQL query and automatically retries
        on certain types of failures (e.g., transient network issues).
        
        Args:
            user_question: The original question from the user.
            extracted_metadata: The metadata from the extraction pipeline.
            max_retries: Maximum number of retry attempts.
            
        Returns:
            Generated SQL query string, or None if all attempts fail.
        """
        logger.info(
            "Attempting SQL generation with retry",
            max_retries=max_retries
        )
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(
                        "Retrying SQL generation",
                        attempt=attempt,
                        max_retries=max_retries
                    )
                
                query = self.generate_query(user_question, extracted_metadata)
                
                if attempt > 0:
                    logger.info(
                        "SQL generation succeeded on retry",
                        attempt=attempt
                    )
                
                return query
                
            except (LLMConnectionError, LLMResponseError) as e:
                # These errors might be transient, worth retrying
                last_exception = e
                logger.warning(
                    "Retryable error during SQL generation",
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(e)
                )
                
                if attempt >= max_retries:
                    logger.error(
                        "Max retries reached for SQL generation",
                        total_attempts=attempt + 1
                    )
                    break
                    
            except SQLGenerationError as e:
                # These errors are not transient, don't retry
                logger.error(
                    "Non-retryable error during SQL generation",
                    error=str(e)
                )
                raise
        
        # All retries exhausted
        logger.error(
            "SQL generation failed after all retry attempts",
            total_attempts=max_retries + 1
        )
        
        if last_exception:
            raise SQLGenerationError(
                f"Failed to generate SQL query after {max_retries + 1} attempts"
            ) from last_exception
        
        return None