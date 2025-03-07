"""
Utilities for extracting, manipulating and tracking information from URIs.
"""

import re

import owlready2 as owl
import rdflib as rdf

URL_ENCODING_MAP = {
    '%20': '_',     # Space
    '%25': 'perc',  # %
    '%28': '',      # (
    '%29': '',      # )
    '%2B': 'plus',  # +
    '%2C': '-',     # ,
    '%2F': '-',     # /
    '%3A': '-',     # :
    '%5B': '',      # [
    '%5D': ''       # ]
}
# Note: these are the only used encodings we found in the .ttl files of Spider4SPARQL


def remove_url_encodings(stem: str) -> str:
    """Replace URL-encoded substrings with an RDF/XML compatible replacement."""
    for enc, sub in URL_ENCODING_MAP.items():
        stem = re.sub(enc, sub, stem)
    stem = re.sub(r'_+', '_', stem)
    stem = re.sub(r'-+', '-', stem)
    return stem


def split_uri(uri: str | rdf.URIRef, separators: str = '/#') -> tuple[str, str]:
    """Split a URI into its base (namespace) and stem (local name)."""
    if isinstance(uri, rdf.URIRef):
        uri = str(uri)
    match = re.search(r'^(?P<base>.+)['+separators+r'](?P<stem>[^'+separators+r']+)$', uri)
    if not match:
        raise RuntimeError(f"Unexpected error: unable to extract 'base' and 'stem' from URI {uri}")
    return match.group('base'), match.group('stem')


def get_stem_from_uri(uri: str | rdf.URIRef, separators: str = '/#') -> str:
    """Extract and post-process a URI stem (local name)."""
    # Get stem, splitting the URI
    _, stem = split_uri(uri, separators)
    # Replace characters conflicting with the RDF/XML format
    stem = re.sub(r'[;]', '-', stem)
    # Remove URL encodings conflicting with the RDF/XML format
    stem = remove_url_encodings(stem)
    # Replace characters creating annoying escapes in Protegé display name
    stem = re.sub(r'[=]', '-', stem)
    # Avoid preserved names in owlready
    if stem == 'label':
        stem = 'has_label'
    # Return cleaned stem
    return stem


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


class RdfOwlMap:
    def __init__(self):
        """Initialize the bidirectional mapping between RDF URIs and Owlready2 entities."""
        self.rdf2owl: dict[rdf.URIRef | rdf.BNode, owl.Thing] = {}
        self.owl2rdf: dict[owl.Thing, set[rdf.URIRef | rdf.BNode]] = {}

    def add_mapping(self, owl_entity: owl.Thing, rdf_uri: rdf.URIRef | rdf.BNode):
        """Adds a bidirectional mapping between an RDF URI and an Owlready2 entity."""
        if rdf_uri in self.rdf2owl:
            raise ValueError(f"URI {rdf_uri} already exists in 'rdf2owl'.")
        self.rdf2owl[rdf_uri] = owl_entity

        if owl_entity not in self.owl2rdf:
            self.owl2rdf[owl_entity] = set()
        self.owl2rdf[owl_entity].add(rdf_uri)

    def to_owl(self, rdf_uri: rdf.URIRef | rdf.BNode) -> owl.Thing:
        """Retrieves the Owlready2 entity corresponding to an RDF URI."""
        return self.rdf2owl.get(rdf_uri, None)

    def to_rdf(self, owl_entity: owl.Thing) -> set[rdf.URIRef | rdf.BNode]:
        """Retrieves the RDF URIs corresponding to an Owlready2 entity."""
        return self.owl2rdf.get(owl_entity, set())

    def has_mapping(self, owl_entity: owl.Thing, rdf_uri: rdf.URIRef | rdf.BNode) -> bool:
        """Checks if a specific bidirectional mapping exists."""
        return self.rdf2owl.get(rdf_uri) == owl_entity and rdf_uri in self.owl2rdf.get(owl_entity, set())
