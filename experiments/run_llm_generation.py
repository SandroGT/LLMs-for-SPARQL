from contextlib import contextmanager
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import owlready2 as owl
import pandas as pd
from tqdm import tqdm

from agents import SparqlGenerationBasic, SparqlGenerationDetailed, ParsingException
from jena import JenaQuery
from llms import Llama3dot3
from logger import LOGGER
from evaluation.comparison import compare_query_results

# Experiment settings
REPETITIONS = 3
LLM_SETTINGS = {
    'temperature': 0.01,
    'top_p': 0.01,
    'max_new_tokens': 2048
}
PROMPT_TYPES = ['basic', 'detailed']
MAX_ERROR_STORE_LEN = 200

# LLM model
LLM_MODEL_NAME = 'llama-3.3-70b'
LOGGER.info(f'Initializing LLM: {LLM_MODEL_NAME}')
LLM_MODEL = Llama3dot3(70)

# File path settings
SCRIPT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR.joinpath('..', 'datasets').resolve()
TEST_GRAPH_DIR = Path('processed', 'graph', 'dev')
TEST_QUERY_DIR = Path('processed', 'queries', 'dev')
RESULTS_DIR = SCRIPT_DIR.joinpath('runs')
GROUND_TRUTH_PATH = SCRIPT_DIR.joinpath('runs', 'ground_truth.json')


def main():
    """Main function to generate SPARQL queries using LLMs on Spider4SPARQL and compare them to its ground truth."""
    # Force model preload
    LLM_MODEL.chat([{'role': 'user', 'content': 'Hey, how are you?'}], max_new_tokens=1)

    # Load ground truth data
    with GROUND_TRUTH_PATH.open('r', encoding='utf8') as f:
        ground_truth_data = json.load(f)

    # Prepare results file
    results_path = RESULTS_DIR.joinpath(f'{LLM_MODEL_NAME}.json')

    # Initialize SPARQL generation agents
    agents = {
        'basic': SparqlGenerationBasic(LLM_MODEL, llm_settings=LLM_SETTINGS),
        'detailed': SparqlGenerationDetailed(LLM_MODEL, llm_settings=LLM_SETTINGS)
    }

    # Initialize the query engine
    query_engine = JenaQuery()
    query_results = {}

    # Iterate over datasets in the dataset directory
    for repetition in tqdm(range(REPETITIONS), total=REPETITIONS, unit='repetitions'):
        print(file=sys.stderr)

        is_first_iteration = (repetition == 0)
        for dataset_path in sorted(DATASET_DIR.iterdir()):
            if not re.match(r'^[a-z].*', dataset_path.name):
                continue
            dataset_name = dataset_path.stem
            if is_first_iteration:
                query_results[dataset_name] = {}
            LOGGER.info(f"Processing dataset: {dataset_name}")

            # Set paths for graph and query files
            graph_path = dataset_path.joinpath(TEST_GRAPH_DIR)
            query_path = dataset_path.joinpath(TEST_QUERY_DIR)

            cumulative_query_id = 0  # Unique query ID across single repetition

            # Process graphs and their corresponding queries
            for graph_file, query_file in zip(sorted(graph_path.iterdir()), sorted(query_path.iterdir())):
                graph_name = graph_file.stem
                if is_first_iteration:
                    query_results[dataset_name][graph_name] = list()
                LOGGER.info(f" - Processing graph: {dataset_name}/{graph_name}")

                # Load queries
                query_df = pd.read_csv(query_file)

                # Load RDF graph
                ontology_world = owl.World()
                ontology = ontology_world.get_ontology(str(graph_file)).load()
                ontology_metadata = {
                    'base_iri': ontology.base_iri,
                    'classes': set(ontology.classes()),
                    'relations': set(ontology.object_properties()),
                    'attributes': set(ontology.data_properties()),
                }

                # Iterate through query rows
                total_queries = 0
                correct_queries_dict = {pt: 0 for pt in PROMPT_TYPES}
                for row_id, row in (pbar := tqdm(query_df.iterrows(), total=len(query_df), unit='queries')):
                    nl_question, partial_sparql, full_sparql = row

                    # Initialize query storage if first iteration
                    if is_first_iteration:
                        query_entry = {
                            'id': row_id,
                            'cumulative_id': cumulative_query_id,
                            'nl_question': nl_question,
                            'evaluation': {style: list() for style in PROMPT_TYPES}
                        }
                        query_results[dataset_name][graph_name].append(query_entry)
                    else:
                        query_entry = query_results[dataset_name][graph_name][row_id]

                    # Process each prompt style
                    for prompt_style in PROMPT_TYPES:
                        query_log = query_entry['evaluation'][prompt_style]
                        assert len(query_log) == repetition

                        llm_agent = agents[prompt_style]

                        # Generate SPARQL query
                        with compute_time() as elapsed_time:
                            raw_output, generated_query = llm_agent.run(
                                system_data=ontology_metadata,
                                user_data={'question': nl_question},
                                parsing_data=ontology_metadata,
                            )
                        generation_duration = elapsed_time()

                        # Try executing the query
                        query_execution_result = None
                        if not isinstance(generated_query, ParsingException):
                            try:
                                query_execution_result = query_engine.run_query(graph_file, generated_query)
                            except Exception as e:
                                query_execution_result = e

                        # Compare results with ground truth
                        is_query_valid = False
                        if isinstance(query_execution_result, dict):
                            expected_results = ground_truth_data[dataset_name][graph_name][row_id]['results']
                            preserve_order = 'order by' in partial_sparql.lower()
                            is_query_valid = compare_query_results(
                                expected_results, query_execution_result, keep_order=preserve_order
                            )
                            if is_query_valid:
                                correct_queries_dict[prompt_style] += 1

                        # Store evaluation data
                        query_log.append({
                            'repetition': repetition,
                            'generation_time': round(generation_duration, 4),
                            'raw_output': raw_output,
                            'generated_sparql':
                                generated_query if not isinstance(generated_query, ParsingException) else None,
                            'parsing_error':
                                str(generated_query)[:MAX_ERROR_STORE_LEN]
                                if isinstance(generated_query, ParsingException) else None,
                            'execution_error':
                                str(query_execution_result)[:MAX_ERROR_STORE_LEN]
                                if isinstance(query_execution_result, Exception) else None,
                            'is_correct': is_query_valid,
                        })

                    # Update counters
                    total_queries += 1
                    cumulative_query_id += 1

                    # Update progress bar
                    pbar_dict = {'repetition': str(repetition)}
                    pbar_dict |= {
                        f'{pt} accuracy': f'{correct_queries_dict[pt]/total_queries: .2f}'
                        f' ({correct_queries_dict[pt]}/{total_queries})'
                        for pt in PROMPT_TYPES
                    }
                    pbar.set_postfix(pbar_dict)

                # Save intermediate results
                with results_path.open('w', encoding='utf8') as f:
                    json.dump(query_results, f, indent=2)


@contextmanager
def compute_time():
    """Context manager for measuring execution time."""
    start_time = perf_counter()
    yield lambda: perf_counter() - start_time


if __name__ == '__main__':
    main()
