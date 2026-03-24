"""지수 백오프 재시도 데코레이터."""

import functools
import time
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


def with_retry(max_attempts: int = 5, base_delay: float = 2.0, jitter: bool = True):
    import random

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    if jitter:
                        delay *= random.uniform(0.5, 1.5)
                    time.sleep(delay)
        return wrapper  # type: ignore
    return decorator
