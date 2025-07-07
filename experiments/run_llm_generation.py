import time
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import owlready2 as owl
import pandas as pd
from tqdm import tqdm

from agents import SparqlGenerationBasic, SparqlGenerationCoT, SparqlGenerationDetailed, ParsingException
from dbpedia import run_sparql_query
from evaluation.comparison import compare_query_results
from jena import JenaQuery
from llms import OllamaServerLLM
from logger import LOGGER
from timeout import set_timeout

# LLM model
LLM_MODEL_NAME = 'deepseek-r1-qwen-32b'
LOGGER.info(f'Initializing LLM: {LLM_MODEL_NAME}')
LLM_MODEL = OllamaServerLLM('deepseek-r1:32b')

# File path settings
SCRIPT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR.joinpath('..', 'datasets').resolve()
TEST_GRAPH_DIR = Path('processed', 'graph', 'dev')
TEST_QUERY_DIR = Path('processed', 'queries', 'dev')
RESULTS_DIR = SCRIPT_DIR.joinpath('runs')
GROUND_TRUTH_PATH = SCRIPT_DIR.joinpath('runs', 'ground_truth.json')

# Experiment settings
DATASET_NAMES = ['spider4sparql', 'bestiary', 'lcquad']
PROMPT_TYPES = ['basic', 'detailed', 'cot']
REPETITIONS_DICT = {
    'basic': 3,
    'detailed': 3,
    'cot': 1
}
LLM_SETTINGS = {
    'temperature': 0.01,
    'top_p': 0.01,
    'max_new_tokens': 2048
}
REASONING_LLM_SETTINGS = {
    'temperature': 0.01,
    'top_p': 0.01,
    'max_new_tokens': 8 * LLM_SETTINGS['max_new_tokens']
}
MAX_ERROR_STORE_LEN = 200
MAX_QUERY_TIME = 5 * 60


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
        'detailed': SparqlGenerationDetailed(LLM_MODEL, llm_settings=LLM_SETTINGS),
        'cot': SparqlGenerationCoT(LLM_MODEL, llm_settings=REASONING_LLM_SETTINGS)
    }

    # Create a mapping from dataset name to proper query function
    local_jena_engine = JenaQuery()
    query_functions_dict = {
        'bestiary': local_jena_engine.run_query,
        'lcquad': run_sparql_query,
        'spider4sparql': local_jena_engine.run_query,
    }

    query_results = {}

    # Iterate over prompt styles, repetitions, and datasets
    for prompt_style in PROMPT_TYPES:
        LOGGER.info(f"Testing prompt strategy: {prompt_style}")

        llm_agent = agents[prompt_style]
        n_repetitions = REPETITIONS_DICT[prompt_style]
        for repetition in tqdm(range(n_repetitions), total=n_repetitions, unit='repetitions'):
            print(file=sys.stderr)

            for dataset_name in DATASET_NAMES:
                dataset_path = DATASET_DIR.joinpath(dataset_name)
                assert dataset_path.exists()
                if not re.match(r'^[a-z].*', dataset_path.name):
                    continue
                dataset_name = dataset_path.stem
                if dataset_name not in query_results:
                    query_results[dataset_name] = {}
                LOGGER.info(f" - Processing dataset: {dataset_name} (repetition {repetition+1}/{n_repetitions})")
                time.sleep(0.01)

                # Define querying function
                query_fun = query_functions_dict[dataset_name]

                # Set paths for graph and query files
                graph_path = dataset_path.joinpath(TEST_GRAPH_DIR)
                query_path = dataset_path.joinpath(TEST_QUERY_DIR)

                cumulative_query_id = 0  # Unique query ID across single repetition

                # Process graphs and their corresponding queries
                for graph_file, query_file in zip(sorted(graph_path.iterdir()), sorted(query_path.iterdir())):
                    graph_name = graph_file.stem
                    if graph_name not in query_results[dataset_name]:
                        query_results[dataset_name][graph_name] = list()
                    LOGGER.info(f"   - Processing graph: {dataset_name}/{graph_name}")
                    time.sleep(0.01)

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

                        # Initialize query storage if it is the first time processing any of this graph queries
                        if len(query_results[dataset_name][graph_name]) < len(query_df):
                            query_entry = {
                                'id': row_id,
                                'cumulative_id': cumulative_query_id,
                                'nl_question': nl_question,
                                'evaluation': dict()
                            }
                            query_results[dataset_name][graph_name].append(query_entry)
                        else:
                            query_entry = query_results[dataset_name][graph_name][row_id]

                        assert prompt_style not in query_entry['evaluation']
                        query_entry['evaluation'][prompt_style] = list()
                        query_log = query_entry['evaluation'][prompt_style]
                        assert len(query_log) == repetition

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
                                with set_timeout(MAX_QUERY_TIME):
                                    query_execution_result = query_fun(query=generated_query, graph_file=graph_file)
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
