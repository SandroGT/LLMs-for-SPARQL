from __future__ import annotations
from typing import Callable

from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.algebra import translateQuery, Minus, Group, Filter


class QueryCategory:
    """Represents a category of SPARQL queries based on structural properties."""
    DEFAULT_CATEGORIES_DATA: dict[str, Callable[[dict], bool]] = {
        '1 hop': lambda d: d['where_triples'] == 1,
        'filter': lambda d: d['filters'] > 0,
        'group': lambda d: d['groupings'] >= 1,
        'minus/union': lambda d: d['minuses'] > 0 or d['unions'] > 0,
        'all': lambda d: True,
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

    def __repr__(self):
        """Gets an object representation."""
        return self.name

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
    group_count = 0
    filter_count = 0
    minus_count = 0
    union_count = 0

    def recursive_visit(pattern):
        """Recursively traverses the SPARQL algebra tree to extract query statistics."""
        nonlocal where_triples, group_count, filter_count, minus_count, union_count
        if getattr(pattern, 'triples', None) is not None:
            where_triples.extend(pattern.triples)
        if getattr(pattern, 'name', None) is not None:
            if 'group' in pattern.name.lower():
                group_count += 1
            if 'filter' in pattern.name.lower():
                filter_count += 1
            if 'minus' in pattern.name.lower():
                minus_count += 1
            if 'union' in pattern.name.lower():
                union_count += 1
        if getattr(pattern, 'p', None) is not None:
            recursive_visit(pattern.p)
        else:
            count = 1
            while getattr(pattern, f'p{count}', None) is not None:
                recursive_visit(getattr(pattern, f'p{count}'))
                count += 1

    recursive_visit(query_patterns)

    query_stats = {
        'where_triples': len(where_triples),
        'groupings': group_count,
        'filters': filter_count,
        'minuses': minus_count,
        'unions': union_count
    }
    print(query_stats)

    return [category for category in categories if category.matches_stats(query_stats)]
