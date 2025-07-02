# SPDX-FileCopyrightText: 2024-present psimon1 <patrick.simon@bam.de>
#
# SPDX-License-Identifier: MIT
import logging

# Ensure the library doesn't emit logging warnings if no handler is configured
logging.getLogger(__name__).addHandler(logging.NullHandler())

from .ERADistNataf import ERADist, ERANataf, ERACond, ERARosen
from .aBUS_SuS import aBUS_SuS, aBUS_SuS_parallel, ErrorWithData

from .__about__ import __version__

def configure_logging(level=logging.WARNING, handler=None):
    """Configure logging for the parallel_abus library.
    
    Args:
        level: Logging level (default: WARNING)
        handler: Custom handler (default: StreamHandler to stderr)
    """
    logger = logging.getLogger(__name__)
    
    # Remove existing handlers to avoid duplication
    logger.handlers.clear()
    
    if handler is None:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs in parent loggers


def configure_module_logging(module_levels=None, handler=None):
    """Configure different logging levels for different modules.
    
    Args:
        module_levels: Dict mapping module names to logging levels.
                      Use short names like 'aBUS_SuS', 'aCS', or full logger names.
                      Example: {'aBUS_SuS': logging.DEBUG, 'aCS': logging.WARNING}
        handler: Custom handler (default: StreamHandler to stderr)
    """
    if module_levels is None:
        module_levels = {}
    
    # Set up common handler if not provided
    if handler is None:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
    
    # Module name mappings for convenience
    module_mappings = {
        'aBUS_SuS': [
            'parallel_abus.aBUS_SuS.aBUS_SuS',
            'parallel_abus.aBUS_SuS.aBUS_SuS_parallel'
        ],
        'aCS': [
            'parallel_abus.aBUS_SuS.aCS_aBUS',
            'parallel_abus.aBUS_SuS.aCS_aBUS_parallel'
        ]
    }
    
    # Configure each module's logging
    for module_key, level in module_levels.items():
        # Check if it's a short name or full logger name
        if module_key in module_mappings:
            # Short name - configure all related loggers
            for logger_name in module_mappings[module_key]:
                logger = logging.getLogger(logger_name)
                logger.handlers.clear()
                logger.addHandler(handler)
                logger.setLevel(level)
                logger.propagate = False
        else:
            # Assume it's a full logger name
            logger = logging.getLogger(module_key)
            logger.handlers.clear()
            logger.addHandler(handler)
            logger.setLevel(level)
            logger.propagate = False


def disable_logging():
    """Disable all logging from the parallel_abus library."""
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
