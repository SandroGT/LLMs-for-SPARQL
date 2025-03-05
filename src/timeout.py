import signal
from contextlib import contextmanager

__all__ = ['TimeoutException', 'set_timeout']


class TimeoutException(Exception):
    pass


def _handle_timeout(signum, frame):
    raise TimeoutException("Operation timed out")


@contextmanager
def set_timeout(seconds):
    # Set the signal handler for SIGALRM
    signal.signal(signal.SIGALRM, _handle_timeout)
    # Schedule an alarm
    signal.alarm(seconds)
    try:
        yield
    except TimeoutException:
        raise
    finally:
        # Cancel the alarm
        signal.alarm(0)
