import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from jena import JenaQuery
from logger import LOGGER

# Paths related to the script and datasets
SCRIPT_PATH = Path(__file__).parent.resolve()
DATASETS_FOLDER = SCRIPT_PATH.joinpath('..', 'datasets').resolve()
TEST_GRAPH_SUB_PATH = Path('processed', 'graph', 'dev')
TEST_QUERY_SUB_PATH = Path('processed', 'queries', 'dev')
RESULTS_FILE = SCRIPT_PATH.joinpath('results', 'ground_truth.json')


def main():
    """Main function to run SPARQL queries on datasets and store the results in a ground truth file.

    The function:
        - Iterates over the datasets in the datasets folder.
        - For each dataset, it reads the corresponding SPARQL queries and graphs.
        - It runs the queries on the respective graphs and stores the results.
        - It saves the results to a JSON file for future reference.

    The ground truth data consists of query results mapped to dataset/graph combinations."""
    # Initialize the query engine and dictionary to store all queries and their results
    query_engine = JenaQuery()
    all_queries = dict()

    # Loop over datasets in the datasets folder (alphabetically sorted)
    for dataset_folder in sorted(f for f in DATASETS_FOLDER.iterdir() if re.match(r'^[a-z].*', f.name)):
        dataset = dataset_folder.stem
        all_queries[dataset] = dict()  # Initialize the dataset key in the results dictionary
        LOGGER.info(f"Checking dataset {dataset}.")

        # Paths to the graph and query files for the dataset
        graph_folder = dataset_folder.joinpath(TEST_GRAPH_SUB_PATH)
        query_folder = dataset_folder.joinpath(TEST_QUERY_SUB_PATH)

        cumulative_query_id = 0  # Initialize the cumulative query ID (within the dataset)

        # Iterate through the graphs and queries in the dataset (one-to-one correspondence)
        for graph_file, query_file in zip(sorted(graph_folder.iterdir()), sorted(query_folder.iterdir())):
            graph = graph_file.stem
            all_queries[dataset][graph] = list()  # Initialize the graph key in the dataset dictionary
            LOGGER.info(f" - Querying graph {dataset}/{graph}.")

            # Load the SPARQL queries from the CSV file
            query_df = pd.read_csv(query_file)

            # Iterate through the rows in the query dataframe
            for row_id, row in tqdm(query_df.iterrows(), total=len(query_df)):
                # Extract the NL question, partial URI, and complete URI from the row
                (nl_question, sparql_partial_uri, sparql_complete_uri) = row

                # Run the query on the current graph and store the results
                all_queries[dataset][graph].append({
                    'id': row_id,  # ID within the graph, not the entire dataset
                    'cumulative_id': cumulative_query_id,  # Unique ID within the dataset
                    'nl_question': nl_question,  # The natural language question
                    'sparql_partial_uri': sparql_partial_uri,  # The partial SPARQL query URI
                    'sparql_complete_uri': sparql_complete_uri,  # The complete SPARQL query URI
                    'results': query_engine.run_query(graph_file, sparql_partial_uri)  # Query result
                })
                cumulative_query_id += 1  # Increment the cumulative query ID

    # Save all queries and results into a JSON file for future use
    with RESULTS_FILE.open('w', encoding='utf8') as f:
        json.dump(all_queries, f, indent=2)


if __name__ == '__main__':
    main()
