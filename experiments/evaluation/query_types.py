from __future__ import annotations
from typing import Callable

from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.algebra import translateQuery


class QueryCategory:
    """Represents a category of SPARQL queries based on structural properties."""
    DEFAULT_CATEGORIES_DATA: dict[str, Callable[[dict], bool]] = {
        "all": lambda d: True,
        "=1 hop": lambda d: d["where_triples"] == 1,
        ">1 hop": lambda d: d["where_triples"] >= 2,
        "=1 filter": lambda d: d["filters"] == 1,
        ">1 filter": lambda d: d["filters"] >= 2,
        ">0 aggregation": lambda d: d["aggregations"] >= 1,
    }

    def __init__(self, name: str, condition: Callable[[dict], bool]):
        """Initializes a query category.

        Args:
            name (str): The name of the category.
            condition (Callable[[dict], bool]): A function that takes query statistics and returns True if the query
             matches this category.
        """
        self.name = name
        self.condition = condition

    def __eq__(self, other):
        """Compares two query categories."""
        if isinstance(other, QueryCategory):
            return self.name == other.name
        return False

    def __hash__(self):
        """Gets object hash."""
        return self.name.__hash__()

    def matches_stats(self, query_stats: dict) -> bool:
        """Checks if the given query statistics match this category."""
        return self.condition(query_stats)

    @classmethod
    def categories(cls) -> list[QueryCategory]:
        """Returns the predefined set of query categories."""
        return [
            cls(name, condition)
            for name, condition in cls.DEFAULT_CATEGORIES_DATA.items()
        ]


def get_query_categories(sparql_query: str, categories: list[QueryCategory]) -> list[QueryCategory]:
    """Analyzes a SPARQL query and returns the matching query categories."""
    parsed_query = parseQuery(sparql_query)
    algebra = translateQuery(parsed_query)

    query_patterns = algebra.algebra
    where_triples = []
    agg_count = 0
    filter_count = 0

    def recursive_visit(pattern):
        """Recursively traverses the SPARQL algebra tree to extract query statistics."""
        nonlocal agg_count, where_triples, filter_count
        if hasattr(pattern, "triples") and pattern.triples is not None:
            where_triples.extend(pattern.triples)
        if hasattr(pattern, "name"):
            if "group" in pattern.name.lower():
                agg_count += 1
            if "filter" in pattern.name.lower():
                filter_count += 1
        if hasattr(pattern, "p"):
            recursive_visit(pattern.p)

    recursive_visit(query_patterns)

    query_stats = {
        "where_triples": len(where_triples),
        "aggregations": agg_count,
        "filters": filter_count,
    }

    return [category for category in categories if category.matches_stats(query_stats)]
