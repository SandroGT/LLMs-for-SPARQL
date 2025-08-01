import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sys
import time
from time import perf_counter

import owlready2 as owl
import pandas as pd
from tqdm import tqdm

from agents import SparqlGenerationBasic, SparqlGenerationCoT, SparqlGenerationDetailed, ParsingException
from dbpedia import run_sparql_query
from evaluation.comparison import compare_query_results
from jena import JenaQuery
from llms import GPTFamilyLLM, Llama3dot1, Llama3dot3, MistralFamilyLLM, OllamaServerLLM, TransformersLLM
from logger import LOGGER
from timeout import set_timeout

# LLM models
LLMS_MAP = {
    'gpt-3.5-turbo': {'code': 'gpt-3.5-turbo-0125', 'class': GPTFamilyLLM},
    'gpt-4o-mini': {'code': 'gpt-4o-mini-2024-07-18', 'class': GPTFamilyLLM},
    'gpt-4o': {'code': 'gpt-4o-2024-08-06', 'class': GPTFamilyLLM},
    'llama-3.1-8b': {'code': 8, 'class': Llama3dot1},
    'llama-3.3-70b': {'code': 70, 'class': Llama3dot3},
    'phi-4-14b': {'code': 'microsoft/phi-4', 'class': TransformersLLM},
    'c4ai-command-r-7b': {'code': 'CohereForAI/c4ai-command-r7b-12-2024', 'class': TransformersLLM},
    'c4ai-command-r-32b': {'code': 'CohereForAI/c4ai-command-r-08-2024', 'class': TransformersLLM},
    'codestral-v0.1-22b': {'code': 'Codestral-22B-v0.1', 'class': MistralFamilyLLM},
    'mistral-small-24b': {'code': 'mistralai/Mistral-Small-24B-Instruct-2501', 'class': TransformersLLM},
    'qwen-2.5-32b': {'code': 'Qwen/Qwen2.5-32B-Instruct', 'class': TransformersLLM},
    'qwen-2.5-coder-32b': {'code': 'Qwen/Qwen2.5-Coder-32B-Instruct', 'class': TransformersLLM},
    'deepseek-v2-coder-16b': {'code': 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', 'class': TransformersLLM},
    'deepseek-r1-qwen-32b': {'code': 'deepseek-r1:32b', 'class': OllamaServerLLM},
}

# File path settings
SCRIPT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR.joinpath('..', 'datasets').resolve()
TEST_GRAPH_DIR = Path('processed', 'graph', 'dev')
TEST_QUERY_DIR = Path('processed', 'queries', 'dev')
RESULTS_DIR = SCRIPT_DIR.joinpath('runs')
GROUND_TRUTH_PATH = SCRIPT_DIR.joinpath('runs', 'ground_truth_lcquad.json')  # TODO FIX

# Experiment settings
DATASET_NAMES = ['lcquad']  # ['spider4sparql', 'bestiary', 'lcquad']  # TODO FIX
PROMPT_TYPES = ['basic', 'detailed', 'cot']
REPETITIONS_DICT = {
    'basic': 3,
    'detailed': 3,
    'cot': 1
}
MAX_ERROR_STORE_LEN = 200
MAX_QUERY_TIME = 5 * 60


def main(llm_name: str):
    """Main function to generate SPARQL queries using LLMs on Spider4SPARQL and compare them to its ground truth."""

    # Loading LLM
    llm_dict = LLMS_MAP[llm_name]
    llm_code, llm_class = llm_dict['code'], llm_dict['class']
    LOGGER.info(f'Initializing LLM: {llm_name}')
    llm_model = llm_class(llm_code)
    llm_model.chat([{'role': 'user', 'content': 'Hey, how are you?'}], max_new_tokens=1)  # Force preload

    # Load ground truth data
    with GROUND_TRUTH_PATH.open('r', encoding='utf8') as f:
        ground_truth_data = json.load(f)

    # Prepare results file
    results_path = RESULTS_DIR.joinpath(f'{llm_name}_lcquad.json')

    # Initialize SPARQL generation agents
    max_new_tokens = 2048
    max_new_tokens_reasoning = (8 if llm_name != 'gpt-3.5-turbo' else 2) * max_new_tokens
    llm_settings = {'temperature': 0.01, 'top_p': 0.01, 'max_new_tokens': max_new_tokens}
    reasoning_llm_settings = {'temperature': 0.01, 'top_p': 0.01, 'max_new_tokens': max_new_tokens_reasoning}
    agents = {
        'basic': SparqlGenerationBasic(llm_model, llm_settings=llm_settings),
        'detailed': SparqlGenerationDetailed(llm_model, llm_settings=llm_settings),
        'cot': SparqlGenerationCoT(llm_model, llm_settings=reasoning_llm_settings)
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
                total_queries = 0
                correct_queries = 0
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
                    for row_id, row in (pbar := tqdm(query_df.iterrows(), total=len(query_df), unit='queries')):
                        nl_question, _, full_sparql = row

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

                        if prompt_style not in query_entry['evaluation']:
                            query_log = list()
                            query_entry['evaluation'][prompt_style] = query_log
                        else:
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
                                    query_execution_result = query_fun(query=generated_query, graph_path=graph_file)
                            except Exception as e:
                                query_execution_result = e

                        # Compare results with ground truth
                        is_query_valid = False
                        if isinstance(query_execution_result, dict):
                            expected_results = ground_truth_data[dataset_name][graph_name][row_id]['results']
                            preserve_order = 'order by' in full_sparql.lower()
                            is_query_valid = compare_query_results(
                                expected_results, query_execution_result, keep_order=preserve_order
                            )
                            if is_query_valid:
                                correct_queries += 1

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
                        pbar_dict = {
                            'repetition': str(repetition),
                            f'{prompt_style} accuracy': f'{correct_queries/total_queries: .2f}'
                            f' ({correct_queries}/{total_queries})'
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--llm',
        required=True,
        choices=LLMS_MAP.keys(),
        help='Select the LLM model to use.'
    )
    args = parser.parse_args()
    main(args.llm)
