"""
Utilities for managing RDFLib knowledge graphs.

This module provides functions to:
- Load and manipulate RDFLib graphs.
- Extract ontology entities (classes, properties, and individuals) from the graphs.
- Infer domain and range for entities.

This implementation assumes that all entities belong to a fixed namespace `RDF_NAMESPACE` used by the dataset.
"""

from pathlib import Path
import re

import rdflib as rdf

RDF_NAMESPACE = rdf.Namespace('http://valuenet/ontop/')
RDF_TYPE = rdf.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')


def load_ttl_graph(ttl_file: Path) -> rdf.Graph:
    """Reads a TTL file, cleans up inconsistencies in the TTL content, and parses it into an RDF graph."""
    # Read the TTL file content
    with ttl_file.open('r', encoding='utf8') as f:
        ttl_text = f.read()
    cleaned_text = re.sub(r'(%20)+(?=>)', '', ttl_text)  # Many URIs have trailing spaces

    # Parse the cleaned text into an RDF graph
    graph = rdf.Graph()
    graph.parse(data=cleaned_text, format='turtle')

    return graph


def get_rdf_classes(graph: rdf.Graph) -> set[rdf.URIRef]:
    """Retrieve all classes in the RDFlib graph."""
    classes = {
        o for s, p, o in graph.triples((None, RDF_TYPE, None))
        if isinstance(o, rdf.URIRef)
    }
    return classes


def get_rdf_object_properties(graph: rdf.Graph) -> set[rdf.URIRef]:
    """Infer object properties from the RDFlib graph.
    Object properties are predicates (p) from the ontology namespace where the object (o) is an individual."""
    object_properties = {
        p for s, p, o in graph.triples((None, None, None))
        if ((isinstance(o, rdf.URIRef) or isinstance(o, rdf.BNode)) and RDF_NAMESPACE in p)
    }
    return object_properties


def get_rdf_data_properties(graph: rdf.Graph) -> set[rdf.URIRef]:
    """Infer data properties from the RDFlib graph.
    Data properties are predicates (p) from the ontology namespace where the object (o) is a raw value (literal)."""
    data_properties = {
        p for s, p, o in graph.triples((None, None, None))
        if (isinstance(o, rdf.Literal) and RDF_NAMESPACE in p)
    }
    return data_properties


def get_rdf_individuals(graph: rdf.Graph) -> tuple[set[rdf.URIRef], set[rdf.BNode]]:
    """Retrieve all individuals (both named and anonymous) in the RDFlib graph."""
    named_individuals = {
        s for s, p, o in graph.triples((None, RDF_TYPE, None))
        if isinstance(s, rdf.URIRef)
    }
    anonymous_individuals = {
        s for s, p, o in graph.triples((None, RDF_TYPE, None))
        if isinstance(s, rdf.BNode)
    }
    return named_individuals, anonymous_individuals


def get_rdf_domain(graph: rdf.Graph, rdf_property: rdf.URIRef) -> set[rdf.URIRef]:
    """Infer the domain of a property in an RDFlib graph."""
    s_set = {s for s, p, o in graph.triples((None, rdf_property, None))}
    domain_ = {t for s in s_set for _, _, t in graph.triples((s, RDF_TYPE, None)) if isinstance(t, rdf.URIRef)}
    return domain_


def get_rdf_range(graph: rdf.Graph, rdf_property: rdf.URIRef) -> set[rdf.URIRef | rdf.Literal]:
    """Infer the range of a property in an RDFlib graph."""
    o_set = {o for s, p, o in graph.triples((None, rdf_property, None))}
    range_ = {t for o in o_set for _, _, t in graph.triples((o, RDF_TYPE, None)) if (isinstance(t, rdf.URIRef))}
    if not range_:
        assert all({isinstance(o, rdf.Literal) for o in o_set}), (rdf_property, o_set)
        range_ = {type(o.value) for o in o_set}
    return range_
