RES_LEN_LIMIT = 5000
"""Maximum number of query results to process before applying heuristic comparison"""


def compare_query_results(
        truth_results_str: list[tuple[str, ...]],
        generated_results_str: list[tuple[str, ...]],
        keep_order: bool = False,
        same_vars: bool = False,
) -> bool:
    """Compare two sets of query results to determine if they match based on specified criteria.

    Args:
        truth_results_str (list[tuple[str, ...]]): The expected query results.
        generated_results_str (list[tuple[str, ...]]): The generated query results to compare.
        keep_order (bool): If True, the order of results matters; otherwise, order is ignored.
        same_vars (bool): If True, both results must have the same variables; otherwise, extra variables are allowed.

    Returns:
        bool: True if the results match based on the given criteria, False otherwise.
    """
    # If the number of results exceeds a predefined limit, assume equality if their lengths match.
    if len(truth_results_str) > RES_LEN_LIMIT or len(generated_results_str) > RES_LEN_LIMIT:
        if len(truth_results_str) == len(generated_results_str):
            return len(generated_results_str[0]) == len(truth_results_str[0]) if same_vars else len(
                generated_results_str[0]) >= len(truth_results_str[0])
        return False

    # Define a comparison function based on whether variables must be the same.
    if same_vars:
        def compare_line(_truth_line: set, _generated_line: set) -> bool:
            return _truth_line == _generated_line  # Must match exactly
    else:
        def compare_line(_truth_line: set, _generated_line: set) -> bool:
            return _truth_line == _truth_line.intersection(_generated_line)  # Allow extra variables

    # Convert tuples to sets to allow for unordered comparison of variables.
    truth_results_str = [set(t) for t in truth_results_str]
    generated_results_str = [set(t) for t in generated_results_str]

    if len(truth_results_str) == len(generated_results_str):
        if not truth_results_str:
            return True  # Both empty

        if keep_order:
            # Compare corresponding elements when order matters
            return all(compare_line(t, g) for t, g in zip(truth_results_str, generated_results_str))
        else:
            # Order-independent comparison; ensure each truth result has a matching generated result
            return all(any(compare_line(t, g) for g in generated_results_str) for t in truth_results_str)

    return False  # Different lengths mean results are not equal


def get_most_voted_result(results_iterations: list) -> tuple[list, int, list] | tuple[list, None, None]:
    """Determine the most frequently occurring result among multiple iterations.

    Args:
        results_iterations (list): A list of query result iterations to compare.

    Returns:
        tuple[list, int, list] | tuple[list, None, None]:
            - A list of index sets, each representing a group of equal results.
            - The index of the most common result if a clear winner exists; otherwise, None.
            - The most common result itself if a clear winner exists; otherwise, None.
    """
    index_sets = []  # List to hold sets of grouped indices
    grouped_indices = set()  # Track indices that have been processed

    # Iterate through results to group equal ones
    for i, result_i in enumerate(results_iterations):
        if i in grouped_indices:
            continue

        current_set = {i}  # Start a new group with the current index

        for j, result_j in enumerate(results_iterations):
            if j != i and j not in grouped_indices:
                equal = (result_i is None and result_j is None) or (
                    result_i is not None and result_j is not None and
                    compare_query_results(result_i, result_j, keep_order=False, same_vars=True)
                )
                if equal:
                    current_set.add(j)

        grouped_indices.update(current_set)
        index_sets.append(current_set)

    # Sort groups by size in descending order
    index_sets.sort(key=len, reverse=True)

    assert index_sets  # Ensure there is at least one group

    if len(index_sets) == 1 or len(index_sets[0]) > len(index_sets[1]):
        valid_result_index = next(iter(index_sets[0]))
        return index_sets, valid_result_index, results_iterations[valid_result_index]
    else:
        return index_sets, None, None


def serialize_jena_results(jena_results_dict: dict) -> list[tuple[str, ...]]:
    """Serializes the results from a Jena query into a structured list of tuples."""
    if 'results' in jena_results_dict:
        # Process SELECT query results
        return [
            tuple(str(var_dict['value']) for var_name, var_dict in record_dict.items())
            for record_dict in jena_results_dict['results']['bindings']
        ]
    elif 'boolean' in jena_results_dict:
        # Process ASK query results
        return [(str(jena_results_dict['boolean']),)]
    else:
        raise ValueError("Unexpected Jena query results format: missing 'bindings' or 'boolean'.")
