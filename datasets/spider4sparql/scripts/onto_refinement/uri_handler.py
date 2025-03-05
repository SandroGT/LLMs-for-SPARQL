"""
Utilities for handling URIs.
"""

import re

NAMESPACE = 'http://valuenet/ontop/'


def name_from_uri(uri: str) -> str:
    """Get an entity name from its URI."""
    return uri[len(NAMESPACE):]


def split_uri(uri: str, separators: str = '/#') -> tuple[str, str]:
    """Split a URI into its base (namespace) and stem (local name)."""
    match = re.search(r'^(?P<base>.+)['+separators+r'](?P<stem>[^'+separators+r']+)$', uri)
    if not match:
        raise RuntimeError(f"Unexpected error: unable to extract 'base' and 'stem' from URI {uri}")
    return match.group('base'), match.group('stem')


def label_from_stem(uri_stem: str, capitalize: bool = True) -> str:
    """Generate a human-readable name from a URI stem."""
    name = uri_stem
    # Remove URL encoding (%<code>)
    name = re.sub(r'%[0-9A-Z]{2}([-_])?', '', name)
    # Format name: handle camelCase and snake_case
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)  # Insert space in camelCase
    name = name.replace('_', ' ')  # Replace underscores with spaces
    if capitalize:
        # Typically for classes
        name = ' '.join(word.capitalize() for word in name.split())  # Capitalize each word
    else:
        # Typically for properties
        name = name.lower()
    return name
