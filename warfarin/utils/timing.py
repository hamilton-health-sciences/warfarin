"""Timer utility for timing individual steps in the pipeline."""

from typing import Optional

import time


class timer:
    """
    Supports timing an individual step in the pipeline, and will output the time
    taken to `stdout`.

    >>> with timer("the name of the step"):
    ...     do_something_that_takes_2_minutes()
    Finished 'the name of the step' in 120.02 seconds
    """
    def __init__(self, name: Optional[str] = None):
        self.name = name

        self.start_time, self.end_time = None, None
        self.total_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args):
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time

        if self.name:
            print(f"Finished '{self.name}' in {self.total_time:.2f} seconds")
