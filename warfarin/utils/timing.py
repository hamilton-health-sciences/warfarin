from typing import Optional

import time


class timer:
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
