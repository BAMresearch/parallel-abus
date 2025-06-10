# SPDX-FileCopyrightText: 2024-present psimon1 <patrick.simon@bam.de>
#
# SPDX-License-Identifier: MIT
import logging

# Ensure the library doesn't emit logging warnings if no handler is configured
logging.getLogger(__name__).addHandler(logging.NullHandler())

from .ERADistNataf import ERADist, ERANataf
from .aBUS_SuS import aBUS_SuS, aBUS_SuS_parallel, ErrorWithData


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


def disable_logging():
    """Disable all logging from the parallel_abus library."""
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
