from evaluation.results import IterationData, QueryExecutionData


def avg_accuracy(results: list[QueryExecutionData | None]) -> float:
    return sum([int(r.is_correct) if r is not None else 0 for r in results]) / len(results)


def avg_generation_time(results: list[QueryExecutionData]) -> float:
    return sum([r.time for r in results]) / len(results)


def avg_parsing_efficiency(results: list[QueryExecutionData]) -> float:
    return sum([1 if r.generation_error is None else 0 for r in results]) / len(results)


def avg_syntax_correctness(results: list[QueryExecutionData]) -> float:
    return sum([1 if r.execution_error is None else 0 for r in results]) / len(results)


def avg_determinism(results: list[IterationData]) -> float:
    return sum([_determinism_score(len(r.groups), len(r.repetitions_data)) for r in results]) / len(results)


def _determinism_score(n_groups: int, n_iters: int) -> float:
    # 1 if deterministic, 0 if all answers are different, linear in-between
    assert n_iters >= 1
    if n_iters == 1:
        return 1
    else:
        return (n_iters - n_groups) / (n_iters-1)
