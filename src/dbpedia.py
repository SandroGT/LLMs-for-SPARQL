import time
from typing import List, Dict

from SPARQLWrapper import SPARQLWrapper, JSON

# Default DBpedia endpoint – replace with your actual constant if defined elsewhere
DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"

# Module-level variable to track last request time
_last_request_time = 0.0


def run_sparql_query(
        query: str,
        endpoint: str = DBPEDIA_ENDPOINT,
        interval: float = 0.1,
        get_bindings: bool = False,
        **_,
) -> dict | list:
    """
    Executes a SPARQL query against a given SPARQL endpoint and returns the result bindings,
    ensuring a minimum interval between requests to avoid overloading the endpoint.

    Args:
        query (str): A SPARQL query string to execute.
        endpoint (str, optional): The SPARQL endpoint URL to query. Defaults to DBpedia's public endpoint.
        interval (float, optional): Minimum time in seconds to wait between two queries. Defaults to 1.0 second.
        get_bindings (bool, optional): Whether to return only the bindings key (True) or the full result dictionary (False).

    Returns:
        List | Dict: A result dictionary or a list of result bindings.
    """
    global _last_request_time
    elapsed = time.time() - _last_request_time

    if elapsed < interval:
        time_to_wait = interval - elapsed
        time.sleep(time_to_wait)

    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        _last_request_time = time.time()
        if get_bindings:
            return results['results']['bindings']
        else:
            return results
    except Exception as e:
        raise e
