import pymysql
import pymysql.cursors
import os
from typing import List, Dict, Any

# Custom modules
from ..logging_config import get_logger
from ..exceptions import DatabaseConnectionError, QueryExecutionError

# --- Logging Configuration ---
logger = get_logger(__name__)

class DatabaseHandler:
    """
    Handles connections to and queries from a MySQL database.
    """
    def __init__(self):
        """
        Initializes the DatabaseHandler by fetching connection details from environment variables.
        
        Raises:
            DatabaseConnectionError: If any required database environment variables are missing.
        """
        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "db": os.getenv("DB_NAME"),
            "port": int(os.getenv("DB_PORT", 13767)),
            "charset": "utf8mb4",
            "cursorclass": pymysql.cursors.DictCursor,
            "connect_timeout": 10,
            "read_timeout": 10,
            "write_timeout": 10,
        }

        # Validate that all required environment variables are set
        required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error("Missing required database environment variables.", missing=missing_vars)
            raise DatabaseConnectionError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        logger.info("DatabaseHandler initialized with provided configuration.")

    def _get_connection(self):
        """Establishes and returns a database connection."""
        try:
            return pymysql.connect(**self.db_config)
        except pymysql.MySQLError as e:
            logger.error("Failed to connect to the database.", error=str(e), exc_info=True)
            raise DatabaseConnectionError("Could not establish a connection to the database.") from e

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes a given SQL query and fetches all results.

        Args:
            query: The SQL query string to be executed.

        Returns:
            A list of dictionaries representing the query results.

        Raises:
            QueryExecutionError: If the query execution fails.
        """
        logger.info("Executing SQL query on the database.")
        logger.debug("Query to execute", sql_query=query)
        
        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                logger.info(f"Query executed successfully, fetched {len(result)} rows.")
                return result
        except pymysql.MySQLError as e:
            logger.error("An error occurred during query execution.", error=str(e), sql_query=query, exc_info=True)
            raise QueryExecutionError("Failed to execute the SQL query on the database.") from e
        finally:
            if connection:
                connection.close()
                logger.info("Database connection closed.")
