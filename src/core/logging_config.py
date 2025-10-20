"""
Logging configuration module for the Metadata Extraction Pipeline.

This module configures structured logging using structlog, providing
context-rich logs suitable for both development and production environments.
"""

import logging
import sys
import os
import structlog


def setup_logging():
    """
    Configures structured logging for the application.
    
    This setup uses `structlog` to provide context-rich, structured logs.
    It renders logs in a human-readable format for development and in JSON format
    for production environments, based on the `APP_ENV` environment variable.
    
    Environment Variables:
        APP_ENV: Application environment ('development' or 'production')
                 Defaults to 'development'
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                   Defaults to 'INFO'
    """
    # Determine the renderer based on the environment.
    # Default to 'development' if APP_ENV is not set.
    app_env = os.getenv("APP_ENV", "development").lower()
    
    if app_env == "production":
        # JSON logs are best for production environments (e.g., Datadog, Splunk).
        renderer = structlog.processors.JSONRenderer()
    else:
        # ConsoleRenderer is human-readable and great for local development.
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    
    # Configure the structlog processors. These add context to every log entry.
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure the standard library logging to work with structlog.
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Validate log level
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level not in valid_levels:
        log_level = "INFO"
        print(f"Warning: Invalid LOG_LEVEL, defaulting to INFO")
    
    root_logger = logging.getLogger()
    
    # Clear any existing handlers to avoid duplicate logs.
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    handler = logging.StreamHandler(sys.stdout)
    # Note: We don't need a formatter here because structlog handles it.
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    
    print(f"Logging setup complete. Environment: '{app_env}', Level: '{log_level}'")


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Returns a logger instance for a given module name.
    
    Args:
        name: The name of the logger, typically __name__.
    
    Returns:
        A structlog bound logger instance with the given name.
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started", version="1.0.0")
    """
    return structlog.get_logger(name)