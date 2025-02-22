

def compare_query_results(
        truth_res: dict,
        pred_res: dict,
        keep_order: bool = False,
        same_vars: bool = False,
        res_len_limit: int = 5000,
) -> bool:
    """Compare two sets of query results to determine if they match based on specified criteria.

    Args:
        truth_res (list[tuple[str, ...]]): The expected query results.
        pred_res (list[tuple[str, ...]]): The generated query results to compare.
        keep_order (bool): If True, the order of results matters; otherwise, order is ignored.
        same_vars (bool): If True, both results must have the same variables; otherwise, extra variables are allowed.
        res_len_limit (int): Maximum number of query results to process before applying heuristic comparison.

    Returns:
        bool: True if the results match based on the given criteria, False otherwise.
    """
    # Serialize results to a structured list of tuples of strings for easier comparison
    serialized_truth_res = serialize_jena_results(truth_res)
    serialized_pred_res = serialize_jena_results(pred_res)
    # If the number of results exceeds a predefined limit, assume equality if their lengths match.
    if len(serialized_truth_res) > res_len_limit or len(serialized_pred_res) > res_len_limit:
        if len(serialized_truth_res) == len(serialized_pred_res):
            if same_vars:
                return len(serialized_pred_res[0]) == len(serialized_truth_res[0])
            else:
                return len(serialized_pred_res[0]) >= len(serialized_truth_res[0])
        return False

    # Define a comparison function based on whether variables must be the same.
    if same_vars:
        def compare_line(_truth_line: set, _generated_line: set) -> bool:
            return _truth_line == _generated_line  # Must match exactly
    else:
        def compare_line(_truth_line: set, _generated_line: set) -> bool:
            return _truth_line == _truth_line.intersection(_generated_line)  # Allow extra variables

    # Convert tuples to sets to allow for unordered comparison of variables.
    serialized_truth_res = [set(t) for t in serialized_truth_res]
    serialized_pred_res = [set(t) for t in serialized_pred_res]

    if len(serialized_truth_res) == len(serialized_pred_res):
        if not serialized_truth_res:
            return True  # Both empty

        if keep_order:
            # Compare corresponding elements when order matters
            return all(compare_line(t, g) for t, g in zip(serialized_truth_res, serialized_pred_res))
        else:
            # Order-independent comparison; ensure each truth result has a matching generated result
            return all(any(compare_line(t, g) for g in serialized_pred_res) for t in serialized_truth_res)

    return False  # Different lengths mean results are not equal


def get_most_voted_result(
        results_iterations: list, res_len_limit: int = 100
) -> tuple[list, int, list] | tuple[list, None, None]:
    """Determine the most frequently occurring result among multiple iterations.

    Args:
        results_iterations (list): A list of query result iterations to compare.
        res_len_limit (int): Maximum number of query results to process before applying heuristic comparison.

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
                    compare_query_results(
                        result_i, result_j, keep_order=False, same_vars=True, res_len_limit=res_len_limit
                    )
                )
                if equal:
                    current_set.add(j)

        grouped_indices.update(current_set)
        index_sets.append(current_set)

    # Sort groups by size in descending order
    index_sets.sort(key=len, reverse=True)

    assert index_sets  # Ensure there is at least one group

    if len(index_sets) == 1 or len(index_sets[0]) > len(index_sets[1]):
        # If there is a most populous group
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
