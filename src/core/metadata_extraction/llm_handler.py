import os
import json
from typing import List, Dict, Optional

# LangChain components for building the pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Import custom logging and exceptions
from ..logging_config import get_logger
from ..exceptions import (
    APIKeyNotFoundError,
    LLMConnectionError,
    LLMResponseError
)

load_dotenv()

# Initialize logger for this module
logger = get_logger(__name__)


class ItemList(BaseModel):
    """A Pydantic model to structure the LLM's output."""
    items: List[str] = Field(description="A list of relevant schema or table names.")


class LLMHandler:
    """
    Handles interactions with a Large Language Model using LangChain.
    
    This class provides methods to interact with Google's Gemini model
    for schema and table selection based on natural language queries.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        """
        Initializes the LangChain-based LLM client.
        
        Args:
            model_name: The name of the Gemini model to use.
        
        Raises:
            APIKeyNotFoundError: If the GOOGLE_API_KEY environment variable is not set.
            LLMConnectionError: If there's an error initializing the model.
        """
        logger.info("Initializing LLMHandler", model_name=model_name)
        
        # Validate API key exists
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY environment variable not found")
            raise APIKeyNotFoundError(
                "GOOGLE_API_KEY environment variable not found. "
                "Please set it in your .env file or environment."
            )
        
        self.model_name = model_name
        
        try:
            # Initialize the ChatGoogleGenerativeAI model with JSON mode
            self.model = ChatGoogleGenerativeAI(
                model=self.model_name,
                convert_system_message_to_human=True,
                temperature=0.1,
                response_mime_type="application/json"
            )
            self.parser = JsonOutputParser(pydantic_object=ItemList)
            
            logger.info(
                "LLMHandler initialized successfully",
                model_name=self.model_name,
                temperature=0.01
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize LLM model",
                model_name=model_name,
                error=str(e),
                exc_info=True
            )
            raise LLMConnectionError(
                f"Failed to initialize LLM model '{model_name}': {str(e)}"
            ) from e

    def _create_chain(self, template: str):
        """
        Helper to create a LangChain pipeline with prompt, model, and parser.
        
        Args:
            template: The prompt template string.
            
        Returns:
            A LangChain pipeline (LCEL chain).
        """
        try:
            prompt = ChatPromptTemplate.from_template(
                template=template,
                partial_variables={
                    "format_instructions": self.parser.get_format_instructions()
                }
            )
            chain = prompt | self.model | self.parser
            logger.debug("LangChain pipeline created successfully")
            return chain
            
        except Exception as e:
            logger.error(
                "Failed to create LangChain pipeline",
                error=str(e),
                exc_info=True
            )
            raise LLMConnectionError(
                f"Failed to create LangChain pipeline: {str(e)}"
            ) from e

    def select_relevant_schemas(
        self,
        user_question: str,
        schema_descriptions: Dict[str, str]
    ) -> List[str]:
        """
        Uses a LangChain pipeline to select relevant schemas based on user question.
        
        Args:
            user_question: The natural language question from the user.
            schema_descriptions: Dictionary mapping schema names to descriptions.
        
        Returns:
            List of relevant schema names.
            
        Raises:
            LLMConnectionError: If there's a network or API connection issue.
            LLMResponseError: If the LLM response is invalid or cannot be parsed.
        """
        logger.info(
            "Selecting relevant schemas",
            user_question=user_question,
            num_schemas=len(schema_descriptions)
        )
        
        template = """
        You are an expert at understanding database structures.
        Based on the user's question, identify the schemas that are most likely to contain the required data.

        User Question:
        {user_question}

        Available Schema Descriptions:
        {schema_descriptions}

        {format_instructions}
        """
        
        try:
            chain = self._create_chain(template)
            
            response = chain.invoke({
                "user_question": user_question,
                "schema_descriptions": json.dumps(schema_descriptions, indent=2)
            })
            
            # Validate response structure
            if not isinstance(response, dict) or "items" not in response:
                logger.error(
                    "Invalid response structure from LLM",
                    response=response
                )
                raise LLMResponseError(
                    "LLM response missing 'items' key or invalid format"
                )
            
            relevant_schemas = response.get("items", [])
            
            logger.info(
                "Relevant schemas identified successfully",
                relevant_schemas=relevant_schemas,
                count=len(relevant_schemas)
            )
            
            return relevant_schemas
            
        except ValidationError as e:
            logger.error(
                "Pydantic validation error during schema selection",
                error=str(e),
                exc_info=True
            )
            raise LLMResponseError(
                f"Failed to validate LLM response: {str(e)}"
            ) from e
            
        except LLMConnectionError:
            # Re-raise connection errors as-is
            raise
            
        except Exception as e:
            logger.error(
                "Unexpected error during schema selection",
                user_question=user_question,
                error=str(e),
                exc_info=True
            )
            raise LLMConnectionError(
                f"Unexpected error during schema selection: {str(e)}"
            ) from e

    def select_relevant_tables(
        self,
        user_question: str,
        table_descriptions: Dict[str, str]
    ) -> List[str]:
        """
        Uses a LangChain pipeline to select relevant tables from a schema.
        
        Args:
            user_question: The natural language question from the user.
            table_descriptions: Dictionary mapping table names to descriptions.
        
        Returns:
            List of relevant table names.
            
        Raises:
            LLMConnectionError: If there's a network or API connection issue.
            LLMResponseError: If the LLM response is invalid or cannot be parsed.
        """
        logger.info(
            "Selecting relevant tables",
            user_question=user_question,
            num_tables=len(table_descriptions)
        )
        
        template = """
        You are an expert at understanding database structures.
        From the given list of tables within a specific schema, identify which tables are needed to answer the user's question.

        User Question:
        {user_question}

        Table Descriptions from the relevant schema:
        {table_descriptions}

        {format_instructions}
        """
        
        try:
            chain = self._create_chain(template)
            
            response = chain.invoke({
                "user_question": user_question,
                "table_descriptions": json.dumps(table_descriptions, indent=2)
            })
            
            # Validate response structure
            if not isinstance(response, dict) or "items" not in response:
                logger.error(
                    "Invalid response structure from LLM",
                    response=response
                )
                raise LLMResponseError(
                    "LLM response missing 'items' key or invalid format"
                )
            
            relevant_tables = response.get("items", [])
            
            logger.info(
                "Relevant tables identified successfully",
                relevant_tables=relevant_tables,
                count=len(relevant_tables)
            )
            
            return relevant_tables
            
        except ValidationError as e:
            logger.error(
                "Pydantic validation error during table selection",
                error=str(e),
                exc_info=True
            )
            raise LLMResponseError(
                f"Failed to validate LLM response: {str(e)}"
            ) from e
            
        except LLMConnectionError:
            # Re-raise connection errors as-is
            raise
            
        except Exception as e:
            logger.error(
                "Unexpected error during table selection",
                user_question=user_question,
                error=str(e),
                exc_info=True
            )
            raise LLMConnectionError(
                f"Unexpected error during table selection: {str(e)}"
            ) from e