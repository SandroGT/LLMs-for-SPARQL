import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dbpedia import run_sparql_query
from evaluation.comparison import serialize_sparql_results
from jena import JenaQuery
from logger import LOGGER

# Paths related to the script and datasets
SCRIPT_PATH = Path(__file__).parent.resolve()
DATASETS_FOLDER = SCRIPT_PATH.joinpath('..', 'datasets').resolve()
DATASET_NAMES = ['spider4sparql', 'bestiary', 'lcquad']
TEST_GRAPH_SUB_PATH = Path('processed', 'graph', 'dev')
TEST_QUERY_SUB_PATH = Path('processed', 'queries', 'dev')
OUTPUT_FILE = SCRIPT_PATH.joinpath('runs', 'ground_truth.json')


def main():
    """Main function to run SPARQL queries on datasets and store the results in a ground truth file.

    The function:
        - Iterates over the datasets in the datasets folder.
        - For each dataset, it reads the corresponding SPARQL queries and graphs.
        - It runs the queries on the respective graphs and stores the results.
        - It saves the results to a JSON file for future reference.

    The ground truth data consists of query results mapped to dataset/graph combinations."""
    # Create a mapping from dataset name to proper query function
    local_jena_engine = JenaQuery()
    query_functions_dict = {
        'bestiary': local_jena_engine.run_query,
        'lcquad': run_sparql_query,
        'spider4sparql': local_jena_engine.run_query,
    }
    all_queries = dict()

    # Loop over datasets in the datasets folder (alphabetically sorted)
    for dataset_name in DATASET_NAMES:
        dataset_folder = DATASETS_FOLDER.joinpath(dataset_name)
        assert dataset_folder.exists() and dataset_folder.is_dir() and dataset_name == dataset_folder.stem
        all_queries[dataset_name] = dict()  # Initialize the dataset key in the results dictionary
        LOGGER.info(f"Checking dataset {dataset_name}.")

        # Define querying function
        query_fun = query_functions_dict[dataset_name]

        # Paths to the graph and query files for the dataset
        graph_folder = dataset_folder.joinpath(TEST_GRAPH_SUB_PATH)
        query_folder = dataset_folder.joinpath(TEST_QUERY_SUB_PATH)

        cumulative_query_id = 0  # Initialize the cumulative query ID (within the dataset)

        # Iterate through the graphs and queries in the dataset (one-to-one correspondence)
        for graph_file, query_file in zip(sorted(graph_folder.iterdir()), sorted(query_folder.iterdir())):
            graph = graph_file.stem
            all_queries[dataset_name][graph] = list()  # Initialize the graph key in the dataset dictionary
            LOGGER.info(f" - Querying graph {dataset_name}/{graph}.")

            # Load the SPARQL queries from the CSV file
            query_df = pd.read_csv(query_file)

            # Iterate through the rows in the query dataframe
            for row_id, row in tqdm(query_df.iterrows(), total=len(query_df)):
                # Extract the NL question, partial URI, and complete URI from the row
                (nl_question, sparql_partial_uri, sparql_complete_uri) = row

                # Run the query on the current graph and store the results
                query_results = query_fun(graph_path=graph_file, query=sparql_complete_uri)
                assert len(serialize_sparql_results(query_results)) > 0
                all_queries[dataset_name][graph].append({
                    'id': row_id,  # ID within the graph, not the entire dataset
                    'cumulative_id': cumulative_query_id,  # Unique ID within the dataset
                    'nl_question': nl_question,  # The natural language question
                    'sparql_partial_uri': sparql_partial_uri,  # The partial SPARQL query URI
                    'sparql_complete_uri': sparql_complete_uri,  # The complete SPARQL query URI
                    'results': query_results  # Query result
                })
                cumulative_query_id += 1  # Increment the cumulative query ID

    # Save all queries and results into a JSON file for future use
    with OUTPUT_FILE.open('w', encoding='utf8') as f:
        json.dump(all_queries, f, indent=2)


if __name__ == '__main__':
    main()
