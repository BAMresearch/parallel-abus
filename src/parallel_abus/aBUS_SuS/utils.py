"""Utils for bust_gemuenden project."""

import time
from typing import Optional


class TimerContext:
    """TimerContext is a context manager for measuring the execution time of code blocks.

    It prints the elapsed time after the context block exits. Custom messages and colorized output can be configured.
    """

    def __init__(self, message: Optional[str] = None, use_color: bool = True):
        self.message = message
        self.use_color = use_color

    def __enter__(self):
        """Start the timer."""
        self._tic = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and print the formatted elapsed time."""
        elapsed_time = time.perf_counter() - self._tic
        formatted_time = self.format_time(elapsed_time)
        msg = f"{self.message} - " if self.message else ""

        print(f"{msg}Elapsed time: {formatted_time}.")

        # Re-raise any exception that occurred in the with block
        if exc_type:
            raise

    @staticmethod
    def format_time(seconds):
        """Format time to display hours, minutes, and seconds as appropriate."""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{int(hours)} h {int(minutes)} min"
        elif minutes:
            return f"{int(minutes)} min {seconds:0.2f} s"
        else:
            return f"{seconds:0.4f} s"
