import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from jena import JenaQuery
from logger import LOGGER
from metrics import compare_query_results, serialize_jena_results


# Paths for dataset and results
SCRIPT_PATH = Path(__file__).parent.resolve()
TARGET_DATASET = 'spider4sparql'
DATASET_FOLDER = SCRIPT_PATH.joinpath('..', 'datasets', TARGET_DATASET).resolve()
GRAPH_SUBPATH = Path('processed', 'graph', 'dev')
QUERY_SUBPATH = Path('processed', 'queries', 'dev')
OUTPUT_FILE = SCRIPT_PATH.joinpath('results', 'sgpt.json')
PREDICTIONS_FILE = SCRIPT_PATH.joinpath('..', 'sgpt', 'outputs', 'spider4sparql_predictions_gpt2-base.json')
GROUND_TRUTH_FILE = SCRIPT_PATH.joinpath('results', 'ground_truth.json')


def main():
    """Main function to compare SGPT evaluation results on Spider4SPARQL with the ground truth."""
    # Load ground truth query results
    with GROUND_TRUTH_FILE.open('r', encoding='utf8') as f:
        ground_truth_results = json.load(f)

    # Load SGPT-generated queries
    with PREDICTIONS_FILE.open('r', encoding='utf8') as f:
        sgpt_predictions = json.load(f)

    # Ensure output file exists
    OUTPUT_FILE.touch()

    # Initialize query execution engine and evaluation storage
    query_engine = JenaQuery()
    evaluation_results = {}

    dataset = TARGET_DATASET
    evaluation_results[dataset] = {}
    LOGGER.info(f"Evaluating dataset: {dataset}")

    # Define paths for dataset's graph and query files
    graph_folder = DATASET_FOLDER.joinpath(GRAPH_SUBPATH)
    query_folder = DATASET_FOLDER.joinpath(QUERY_SUBPATH)

    query_index = 0  # Tracks the global query ID across all graphs

    # Process graphs and corresponding queries in the dataset
    for graph_file, query_file in zip(sorted(graph_folder.iterdir()), sorted(query_folder.iterdir())):
        graph_name = graph_file.stem
        evaluation_results[dataset][graph_name] = list()
        LOGGER.info(f" - Evaluating queries on graph: {dataset}/{graph_name}")

        # Load SPARQL queries from the CSV file
        query_df = pd.read_csv(query_file)

        # Track accuracy statistics
        total_queries = total_correct_queries = 0

        for row_id, row in (pbar := tqdm(query_df.iterrows(), total=len(query_df))):
            # Extract relevant fields from the dataframe row
            nl_question, partial_sparql, complete_sparql = row

            # Initialize evaluation record for this query
            sgpt_eval_results = list()
            query_evaluation = {
                'id': row_id,  # Query index within this graph
                'global_id': query_index,  # Unique query index across the dataset
                'nl_question': nl_question,
                'evaluation': {'sgpt': sgpt_eval_results}
            }
            evaluation_results[dataset][graph_name].append(query_evaluation)

            # Retrieve the corresponding SGPT-predicted query
            sgpt_query_data = sgpt_predictions[query_index]
            sgpt_reference_query = f'PREFIX : <http://valuenet/ontop/>\n{sgpt_query_data["ground_truth_sparql"]}'
            sgpt_predicted_query = f'PREFIX : <http://valuenet/ontop/>\n{sgpt_query_data["predicted_sparql"]}'

            # Ensure that the ground truth queries match
            if not check_query_equality(partial_sparql, sgpt_reference_query):
                raise RuntimeError("Mismatch between dataset ground truth and SGPT reference queries")

            # Execute the SGPT-predicted SPARQL query
            try:
                execution_results = query_engine.run_query(graph_file, sgpt_predicted_query)
            except Exception as e:
                execution_results = e  # Capture query execution errors

            # Compare query execution results with ground truth
            if not isinstance(execution_results, Exception):
                assert isinstance(execution_results, dict)
                expected_results = ground_truth_results[dataset][graph_name][row_id]['results']
                preserve_order = 'order by' in partial_sparql.lower()
                query_matches_ground_truth = compare_query_results(
                    serialize_jena_results(expected_results),
                    serialize_jena_results(execution_results),
                    keep_order=preserve_order
                )
                if query_matches_ground_truth:
                    total_correct_queries += 1
            else:
                query_matches_ground_truth = False

            # Store execution results and correctness evaluation
            sgpt_eval_results.append({
                'execution_error': str(execution_results) if isinstance(execution_results, Exception) else None,
                'execution_results': execution_results if not isinstance(execution_results, Exception) else None,
                'is_correct': query_matches_ground_truth,
            })

            # Update counters
            total_queries += 1
            query_index += 1

            # Save evaluation results incrementally
            with OUTPUT_FILE.open('w', encoding='utf8') as f:
                json.dump(evaluation_results, f, indent=2)

            # Update progress bar with current accuracy
            accuracy = total_correct_queries / total_queries
            pbar.set_postfix({'accuracy': f'{accuracy: .2%} ({total_correct_queries}/{total_queries})'})


def check_query_equality(dataset_gt_query: str, sgpt_gt_query: str) -> bool:
    """Checks if SGPT's stored ground truth query matches the dataset ground truth."""
    # Normalize by removing spaces and converting to lowercase
    dataset_gt_query = dataset_gt_query.replace(' ', '').lower()
    sgpt_gt_query = sgpt_gt_query.replace(' ', '').lower()

    # SGPT's stored ground truth may be truncated due to token limits
    return sgpt_gt_query in dataset_gt_query


if __name__ == '__main__':
    main()
