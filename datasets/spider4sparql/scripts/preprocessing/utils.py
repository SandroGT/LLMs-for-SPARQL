"""
Generic utilities.
"""

import re


def convert_bytes(num) -> str:
    """Convert a file size in bytes to a human-readable format (e.g., KB, MB, GB, TB)."""
    for dim in ['b', 'KB', 'MB', 'GB']:
        if num < 1024.0:
            return f'{num:.2f}{dim}'
        num /= 1024.0
    return f'{num:.2f}TB'


def get_python_name(name: str) -> str:
    """Convert a string into a Python-friendly variable name."""
    clean_name = re.sub(r'[/#\-\s]', '_', name)
    return f'py_{clean_name}'
