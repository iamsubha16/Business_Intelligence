"""
Custom exceptions for the Metadata Extraction Pipeline.

Defining custom exceptions makes error handling more specific and clear,
allowing the application to catch and respond to different error
conditions appropriately.
"""


class BusinessIntvException(Exception):
    """
    Base exception class for this application.
    
    All custom exceptions in this module inherit from this base class,
    allowing for easy catching of all application-specific errors.
    """
    pass


class APIKeyNotFoundError(BusinessIntvException):
    """
    Raised when the GOOGLE_API_KEY environment variable is not set.
    
    This exception indicates that the application cannot initialize
    the LLM client because the required API key is missing.
    """
    pass


class DataLoadError(BusinessIntvException):
    """
    Raised when there is an error loading the metadata file.
    
    This can occur due to:
    - File not found
    - Invalid JSON format
    - Permission issues
    - Corrupt file data
    """
    pass


class LLMConnectionError(BusinessIntvException):
    """
    Raised for issues connecting to the LLM API.
    
    This can occur due to:
    - Network connectivity problems
    - API endpoint unavailability
    - Authentication failures
    - Rate limiting
    - Timeout errors
    """
    pass


class LLMResponseError(BusinessIntvException):
    """
    Raised when the LLM response is invalid, malformed, or fails parsing.
    
    This can occur due to:
    - Unexpected response format
    - Missing required fields
    - Invalid JSON structure
    - Pydantic validation failures
    """
    pass


class SQLGenerationError(BusinessIntvException):
    """
    Raised when SQL query generation fails.
    
    This can occur due to:
    - Invalid or empty metadata structure
    - LLM returning malformed SQL
    - Empty or invalid responses from LLM
    - SQL validation failures
    """
    pass


class DatabaseConnectionError(BusinessIntvException):
    """Raised when a connection to the database cannot be established."""
    pass


class QueryExecutionError(BusinessIntvException):
    """Raised when an error occurs during SQL query execution."""
    pass

