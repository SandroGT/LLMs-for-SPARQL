from evaluation.results import IterationData, QueryExecutionData


def avg_accuracy(results: list[QueryExecutionData | None]) -> float:
    """Calculate the average accuracy, defined as the ratio of correct queries to the total number of queries."""
    if len(results) == 0:
        return 1.0
    return sum([int(r.is_correct) if r is not None else 0 for r in results]) / len(results)


def avg_generation_time(results: list[QueryExecutionData]) -> float:
    """Calculate the average generation time for a list of query execution results."""
    if len(results) == 0:
        return 1.0
    return sum([r.time for r in results]) / len(results)


def avg_parsing_success_rate(results: list[QueryExecutionData]) -> float:
    """Calculate the success rate of parsing the LLM-generated answers, based on whether there was a parsing error.
    This metric does not guarantee that the parsing correctly extracted only the SPARQL query from the answer, but
    measures if parsing succeeded without errors."""
    if len(results) == 0:
        return 1.0
    return sum([1 if r.generation_error is None else 0 for r in results]) / len(results)


def avg_syntax_correctness(results: list[QueryExecutionData]) -> float:
    """Calculate the average syntax correctness of the queries, defined as the proportion of queries that did not
    result in execution errors when run in a Jena engine. Failures may arise because the SPARQL query includes
    errors or because parsing didn't isolate it perfectly. The latter mostly happens due to the LLM not respecting
    the requested format, even when we attempt to parse flexibly."""
    if len(results) == 0:
        return 1.0
    return sum([1 if r.execution_error is None else 0 for r in results]) / len(results)


def avg_determinism(results: list[IterationData]) -> float:
    """Calculate the average determinism of answers to repeated queries. This metric measures how consistent
    the answers are when the same question is asked multiple times."""
    if len(results) == 0:
        return 1.0
    return sum([_determinism_score(len(r.groups), len(r.repetitions_data)) for r in results]) / len(results)


def _determinism_score(n_groups: int, n_iters: int) -> float:
    """Calculate a determinism score based on the number of groups and iterations. A score of 1 indicates perfect
    consistency, while lower values reflect increasing inconsistency in the answers over iterations."""
    assert n_iters >= 1
    if n_iters == 1:
        return 1
    else:
        return (n_iters - n_groups) / (n_iters - 1)
