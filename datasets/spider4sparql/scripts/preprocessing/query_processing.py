"""
Utilities for manipulating, validating and executing queries.
"""

from pathlib import Path
import re

import rdflib as rdf

from jena import JenaQuery
from rdflib_processing import (RDF_NAMESPACE, load_ttl_graph, get_rdf_individuals, get_rdf_classes,
                               get_rdf_object_properties, get_rdf_data_properties)
from uri_manipulation import split_uri, RdfOwlMap

# Initialize the query engine using Jena
QUERY_ENGINE = JenaQuery()


def preprocess_ttl_query(raw_query: str, all_graph_uris: list[rdf.URIRef]) -> str:
    """Preprocesses SPARQL queries by replacing partial URIs with complete URIs.

    Modifies raw SPARQL queries to make them executable by:
    - Replacing partial URIs (e.g., ":<name>") with complete URIs.
    - Cleaning the query by removing extra characters and spaces.
    """
    new_query = raw_query
    new_query = re.sub(r'\\#', '#', new_query)
    new_query = re.sub(r' +', ' ', new_query)

    for uri in all_graph_uris:
        _, stem = split_uri(uri, separators='/')
        # Replace partial URI (":" prefix + URI stem) with complete URI
        new_query = re.sub(rf':{stem}(?=\s)', f'<{uri}>', new_query)

    return new_query.strip()


def convert_to_rdf_query(ttl_complete_query: str, map_obj: RdfOwlMap) -> tuple[str, str]:
    """Converts a TTL compatible query to use Owlready2 entity URIs.

    Takes a SPARQL query and transforms it by:
    1. Replacing RDFlib URIs with the corresponding Owlready2 entity URIs.
    2. Converting complete URIs (in angular brackets) into partial URIs by replacing them with their corresponding
       "stem" part after the RDF namespace and prepending the `:` symbol.
    3. Adding the RDF namespace prefix to the query.

    Return both query with Owlready2 full entity URIs and the query with partial URIs (using `:` for the RDF namespace).
    """
    rdf_complete_query = ttl_complete_query

    # Replace each RDFlib URI in the query with its corresponding Owlready2 entity URI
    for rdf_uri, owl_entity in map_obj.rdf2owl.items():
        rdf_complete_query = re.sub(str(rdf_uri), owl_entity.iri, rdf_complete_query)

    # Replace complete URIs with partial URIs
    prefix = RDF_NAMESPACE
    rdf_partial_query = rdf_complete_query
    for owl_entity in map_obj.owl2rdf.keys():
        # Construct the complete URI of the owl entity
        complete_uri = owl_entity.iri

        # Extract the stem of the URI after the RDF_NAMESPACE
        if complete_uri.startswith(str(prefix)):
            stem = complete_uri[len(str(prefix)):]
            partial_uri = f':{stem}'

            # Replace the complete URI in the query with the partial URI
            rdf_partial_query = re.sub(rf'<{re.escape(str(complete_uri))}>', partial_uri, rdf_partial_query)

    # Add the prefix to the query
    rdf_partial_query = f"PREFIX : <{str(prefix)}>\n{rdf_partial_query}"

    return rdf_complete_query, rdf_partial_query


def check_mentions(ttl_file: Path, queries: list[str]):
    """Checks whether TTL queries reference appropriate RDF entities.

    Validates the following assumptions about the queries:
    1. Queries do not explicitly mention individuals.
    2. Queries do reference at least one RDF class.
    3. Queries do reference at least one RDF property (either object or data properties).
    """
    graph = load_ttl_graph(ttl_file)

    # Verify that queries never explicitly mention individuals.
    # This is an assumption we enforce by checking if any individual is referenced in the queries.
    assert not any(
        {i[len(RDF_NAMESPACE):] in q for ic in get_rdf_individuals(graph) for i in ic for q in queries}
    ), "Queries should not explicitly mention individuals."

    # Ensure that queries reference at least one RDF class.
    assert any(
        {split_uri(c)[1] in q for c in get_rdf_classes(graph) for q in queries}
    ), "Queries must reference at least one class."

    # Ensure that queries reference at least one RDF property (object or data property).
    assert any(
        {split_uri(r)[1] in q for r in get_rdf_object_properties(graph) for q in queries} |
        {split_uri(a)[1] in q for a in get_rdf_data_properties(graph) for q in queries}
    ), "Queries must reference at least one property (object or data property)."


def safe_query(graph: Path, query: str) -> tuple[dict, int, bool, Exception]:
    """Safely executes a SPARQL query on a given graph.

    Runs a SPARQL query on the provided graph using the Jena query engine. It handles exceptions gracefully, returning
    the query runs, their number, and any failure information.
    """
    try:
        # Execute the query using the Jena query engine
        results = QUERY_ENGINE.run_query(graph.resolve(), query)

        # Count the number of result bindings
        num_bindings = len(results['runs']['bindings'])

        # Indicate no failure occurred
        failed = False
        exception = None
    except Exception as e:
        # Handle any exception during query execution
        results = None
        num_bindings = None
        failed = True
        exception = e

    # Return the runs, count, failure flag, and exception
    return results, num_bindings, failed, exception
