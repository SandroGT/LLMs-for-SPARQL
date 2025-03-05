import json
import math
from pathlib import Path
import random
import re
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from logger import LOGGER
from evaluation.comparison import get_most_voted_result
from evaluation.metrics import (avg_accuracy, avg_generation_time, avg_syntax_correctness, avg_determinism)
from evaluation.query_types import QueryCategory, get_query_categories
from evaluation.results import RunResults
from jena import JenaQuery
from timeout import set_timeout, TimeoutException


# === Directory Paths ===
SCRIPT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR.joinpath('..', 'datasets').resolve()

# Processed data paths
TEST_GRAPH_DIR = Path('processed', 'graph', 'dev')
TEST_QUERY_DIR = Path('processed', 'queries', 'dev')

# Evaluation and results storage
RESULTS_DIR = Path('runs')
EVALUATION_DIR = SCRIPT_DIR.joinpath('evaluation')
SCORES_DIR = EVALUATION_DIR.joinpath('scores')
WRONG_QUERIES_PATH = EVALUATION_DIR.joinpath('wrong_queries.json')

# Support data paths
SUPPORT_DIR = EVALUATION_DIR.joinpath('support_data')
GROUPINGS_DATA = SUPPORT_DIR.joinpath('groupings.json')
CATEGORIES_DATA = SUPPORT_DIR.joinpath('categories.json')

# Specific evaluation results
GROUND_TRUTH_PATH = RESULTS_DIR.joinpath('ground_truth.json')
BASELINE_PATH = RESULTS_DIR.joinpath('sgpt.json')

# === Constants ===
GROUPING_MAX_QUERY_TIME = 120
ROUND_SCORES_DIGITS = 4

# Model and prompt settings
LLMS_ORDER = [
    # OpenAI GPT
    'gpt-3.5-turbo',
    'gpt-4o-mini',
    'gpt-4o',

    # Meta Llama
    'llama-3.1-8b',
    'llama-3.3-70b',

    # Microsoft Phi
    'phi-4-14b',

    # Cohere Command R
    'c4ai-command-r-7b',
    'c4ai-command-r-32b',

    # Mistral AI
    'codestral-v0.1-22b',
    'mistral-small-24b',

    # Alibaba Qwen
    'qwen-2.5-32b',
    'qwen-2.5-coder-32b',

    # DeepSeek
    'deepseek-v2-coder-16b',
    'deepseek-r1-qwen-32b'
]

PROMPT_TYPES = ['basic', 'detailed']
ENSEMBLE_NAME = 'ensemble'
DATASETS = ['spider4sparql', 'bestiary']
INCORRECT_ANSWER_THRESHOLD = int(round(1.00 * len(LLMS_ORDER)))
MAX_WRONG_QUERY_SAMPLES = 10

# Color mappings for visualization
# SEE Colors for GPT, Llama, Phi and Cohere: https://coolors.co/ffb885-ffa375-ff8c66-dca984-d89879-ffd37a-bff28c-9bf589
# SEE Colors for Mistral, Qwen and DeepSeek: https://coolors.co/a1f7e2-83fce8-8dd0fc-88bafc-c2bceb-c0aeea
LLMS_COLORS_DICT = {
    'gpt-3.5-turbo': '#FFB885',
    'gpt-4o-mini': '#FFA375',
    'gpt-4o': '#FF8C66',
    'llama-3.1-8b': '#DCA984',
    'llama-3.3-70b': '#D89879',
    'phi-4-14b': '#FFD37A',
    'c4ai-command-r-7b': '#BFF28C',
    'c4ai-command-r-32b': '#9BF589',
    'mistral-small-24b': '#83FCE8',
    'codestral-v0.1-22b': '#A1F7E2',
    'qwen-2.5-32b': '#8DD0FC',
    'qwen-2.5-coder-32b': '#88BAFC',
    'deepseek-v2-coder-16b': '#C2BCEB',
    'deepseek-r1-qwen-32b': '#C0AEEA',
}
BASELINE_COLOR = '#808080'

# Initialize random seed
random.seed(42)


def main():
    """Evaluate generated SPARQL queries by comparing them to the ground truth and computing various metrics."""
    # Create output folders
    if not SCORES_DIR.exists():
        SCORES_DIR.mkdir()
    if not SUPPORT_DIR.exists():
        SUPPORT_DIR.mkdir()

    # Load query categories and categorize ground truth queries
    categories = QueryCategory.categories()
    query_categories = categorize_queries(categories)

    # Load model outputs for evaluation
    sgpt_run_results = load_sgpt_run(query_categories)
    llms_run_dict, llms_run_results = load_llms_run(query_categories)

    # Count the number of queries in each category
    categories_count = {
        category.name: len(sgpt_run_results.retrieve(query_categories=category))
        for category in categories
    }

    # Compute baseline accuracy for SGPT
    sgpt_accuracy_dict = {
        'sgpt': {
            category.name: round(avg_accuracy(sgpt_run_results.retrieve(query_categories=category)), ROUND_SCORES_DIGITS)
            for category in categories
        }
    }

    # Compute LLM accuracy scores for different prompts and categories
    scores_accuracy_dict = get_categorized_score_dict(llms_run_results, categories, avg_accuracy, most_voted=True)

    # Add an ensemble score where LLMs are correct if at least one of the two prompts produces a correct query
    scores_accuracy_dict |= {
        ENSEMBLE_NAME: {
            llm_code: {
                category.name: round(avg_accuracy([
                    b if (b is not None and b.is_correct) else d
                    for b, d in zip(
                        llm_run_result.retrieve(prompts='basic', query_categories=category, most_voted=True),
                        llm_run_result.retrieve(prompts='detailed', query_categories=category, most_voted=True)
                    )
                ]), ROUND_SCORES_DIGITS)
                for category in categories
            }
            for llm_code, llm_run_result in llms_run_results.items()
        }
    }

    # Plot and store accuracy scores
    score_name = 'accuracy'
    for prompt_type, llms_accuracy_dict in scores_accuracy_dict.items():
        plot_category_bars(prompt_type, score_name, categories_count, llms_accuracy_dict, sgpt_accuracy_dict)

    # Compute and store accuracy table
    filename, metric, filters = ('accuracy.csv', avg_accuracy, {'most_voted': True})
    result_dict = {
        dataset: get_categorized_score_dict(llms_run_results, categories, metric, datasets=dataset, **filters)
        for dataset in DATASETS
    }
    for dataset in DATASETS:
        result_dict[dataset] |= {
            ENSEMBLE_NAME: {
                llm_code: {
                    category.name: round(avg_accuracy([
                        b if (b is not None and b.is_correct) else d
                        for b, d in zip(
                            llm_run_result.retrieve(
                                prompts='basic', query_categories=category, datasets=dataset, **filters),
                            llm_run_result.retrieve(
                                prompts='detailed', query_categories=category, datasets=dataset, **filters)
                        )
                    ]), ROUND_SCORES_DIGITS)
                    for category in categories
                }
                for llm_code, llm_run_result in llms_run_results.items()
            }
        }
    table_data = {
        f'{dataset_name}-{prompt_type}': {
            model_code: scores['all']
            for model_code, scores in model_results.items()
        }
        for dataset_name, prompts_dict in result_dict.items()
        for prompt_type, model_results in prompts_dict.items()
    }

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame.from_dict(table_data, orient='index')
    csv_path = SCORES_DIR.joinpath(filename)
    df.to_csv(csv_path, encoding='utf8')

    # Compute and store additional evaluation metrics tables
    scores_data = [
        ('generation_time.csv', avg_generation_time, {'most_voted': False}),
        ('syntax_correctness.csv', avg_syntax_correctness, {'most_voted': False}),
        ('determinism.csv', avg_determinism, {'return_iterations': True}),
    ]
    for filename, metric, filters in scores_data:
        result_dict = get_categorized_score_dict(llms_run_results, categories, metric, datasets=dataset, **filters)
        table_data = {
            prompt_type: {
                model_code: scores['all']
                for model_code, scores in model_results.items()
            }
            for prompt_type, model_results in result_dict.items()
        }

        # Convert to DataFrame and save as CSV
        df = pd.DataFrame.from_dict(table_data, orient='index')
        csv_path = SCORES_DIR.joinpath(filename)
        df.to_csv(csv_path, encoding='utf8')

    # Store a sample of faulty queries
    store_mostly_incorrect_queries(llms_run_results, llms_run_dict)


def load_sgpt_run(query_categories: dict) -> RunResults:
    """Load SGPT run data from a JSON file and map it to a `RunResults` data structure."""
    # Load SGPT run outputs from the specified path
    LOGGER.info("Loading and processing SGPT runs.")
    with BASELINE_PATH.open('r', encoding='utf8') as f:
        sgpt_data = json.load(f)

    # Iterate over the loaded SGPT data and add necessary modifications
    for dataset_name, dataset_dict in sgpt_data.items():
        for graph_name, query_list in dataset_dict.items():
            for query_dict in query_list:
                # Add a fake repetition for consistency with other models (LLMs)
                query_dict['evaluation']['sgpt'] = {
                    'groups': [[0]],  # Single group for SGPT output
                    'most_voted_id': 0,  # Only one repetition, so most voted is the first
                    'repetitions': query_dict['evaluation']['sgpt']  # Use the SGPT evaluation as the repetition data
                }

                # Assign the appropriate categories to each query based on the provided mapping
                try:
                    query_dict['categories'] = query_categories[dataset_name][graph_name][query_dict['id']]
                except KeyError as e:
                    LOGGER.warning(
                        f"Missing category for query {query_dict['id']} in dataset {dataset_name}, graph {graph_name}: {e}")
                    query_dict['categories'] = set()  # Default to an empty set if no category is found

    # Return the structured data as a RunResults object
    return RunResults(sgpt_data)


def load_llms_run(query_categories: dict) -> tuple[dict, dict[str, RunResults]]:
    """Load LLMs run data from a JSON file and map it to a `RunResults` data structure."""
    LOGGER.info("Loading and processing zero-shot LLM runs.")
    llm_data = dict()

    # Load stored LLM-generated queries, skipping ground truth and other reference files
    for file in [RESULTS_DIR.joinpath(f'{name}.json') for name in LLMS_ORDER]:
        with file.open('r', encoding='utf8') as f:
            llm_code = file.stem  # Extracts model identifier from filename
            llm_data[llm_code] = json.load(f)

    # Load existing groupings if available, otherwise initialize an empty structure
    # Groupings represent clusters of queries that yield the same execution results.
    # They help measure LLM consistency by identifying which queries produce identical outputs.
    # The most frequent result (most-voted) serves as a reference for stability assessment.
    if GROUPINGS_DATA.exists():
        with GROUPINGS_DATA.open('r', encoding='utf8') as f:
            groupings_data = json.load(f)
    else:
        groupings_data = dict()

    # Initialize the query execution engine
    query_engine = JenaQuery()

    # Iterate over the LLMs run data to complete it
    any_update = False
    for llm_code, llm_dict in llm_data.items():
        new_groupings_needed = llm_code not in groupings_data
        if new_groupings_needed:
            any_update = True
            groupings_data[llm_code] = dict()

        for dataset_name, dataset_dict in llm_dict.items():
            if new_groupings_needed:
                any_update = True
                groupings_data[llm_code][dataset_name] = dict()

            for graph_name, query_list in dataset_dict.items():
                if new_groupings_needed:
                    any_update = True
                    groupings_data[llm_code][dataset_name][graph_name] = list()
                    LOGGER.info(f'Processing groupings for {llm_code}-{dataset_name}/{graph_name}')
                    queries_pbar = tqdm(enumerate(query_list), total=len(query_list), unit='query')
                else:
                    queries_pbar = enumerate(query_list)

                graph_path = DATASET_DIR.joinpath(dataset_name, TEST_GRAPH_DIR, f'{graph_name}.rdf')

                for query_id, query_dict in queries_pbar:
                    if new_groupings_needed:
                        any_update = True
                        groupings_data[llm_code][dataset_name][graph_name].append(dict())

                        for prompt_type, results_list in query_dict['evaluation'].items():
                            # Execute SPARQL queries unless they previously failed (marked by execution_error)
                            query_results = list()
                            for res in results_list:
                                r = None
                                if res['execution_error'] is None:
                                    try:
                                        # Set a timeout to speed up
                                        with set_timeout(GROUPING_MAX_QUERY_TIME):
                                            r = query_engine.run_query(graph_path, res['generated_sparql'])
                                    except TimeoutException:
                                        pass
                                query_results.append(r)

                            # Compute groupings based on query execution results
                            groups_index_sets, index_most_voted, _ = get_most_voted_result(query_results)
                            groupings_data[llm_code][dataset_name][graph_name][query_id][prompt_type] = {
                                'groups': [list(g) for g in groups_index_sets],
                                'most_voted_id': index_most_voted
                            }

                    # Update query dictionary with groupings
                    for prompt_type in query_dict['evaluation']:
                        results_list = query_dict['evaluation'][prompt_type]
                        query_dict['evaluation'][prompt_type] = {
                            'groups':
                                groupings_data[llm_code][dataset_name][graph_name][query_id][prompt_type]['groups'],
                            'most_voted_id':
                                groupings_data[llm_code][dataset_name][graph_name][query_id][prompt_type]['most_voted_id'],
                            'repetitions': results_list
                        }
                    query_dict['categories'] = query_categories[dataset_name][graph_name][query_dict['id']]

        # Periodic saving to avoid data loss in case of crashes
        if any_update:
            with GROUPINGS_DATA.open('w', encoding='utf8') as f:
                json.dump(groupings_data, f, indent=2)

    # Final save to ensure all groupings are written
    if any_update:
        with GROUPINGS_DATA.open('w', encoding='utf8') as f:
            json.dump(groupings_data, f, indent=2)

    # Return the data
    run_results = {
        llm_code: RunResults(llm_dict)
        for llm_code, llm_dict in llm_data.items()
    }
    return llm_data, run_results


def categorize_queries(
        categories: list[QueryCategory],
        reset: bool = False
) -> dict:
    """Categorizes queries from a ground truth dataset loaded from a JSON file and stores the results in a JSON file.

    Args:
        categories (list[QueryCategory]): List of all query categories.
        reset (bool): If True, forces the recomputation of query categories from scratch. If False, attempts to load
                      previously computed categories from the output file (if it exists).

    Returns:
        dict: A dictionary containing the categorized queries for each dataset and graph.
    """
    # Load ground truth data from the provided path
    LOGGER.info("Loading ground truth data.")
    with GROUND_TRUTH_PATH.open('r', encoding='utf8') as f:
        ground_truth_data = json.load(f)

    # Load category mappings (assuming QueryCategory.categories() returns a dictionary {name: category_obj})
    categories_dict = {c.name: c for c in categories}

    # Load existing ground truth categories if not resetting
    if not reset and CATEGORIES_DATA.exists():
        with CATEGORIES_DATA.open('r', encoding='utf8') as f:
            ground_truth_categories_json = json.load(f)
    else:
        ground_truth_categories_json = {
            dataset_name: {graph_name: [] for graph_name in dataset_dict.keys()}
            for dataset_name, dataset_dict in ground_truth_data.items()
        }

    def iter_queries():
        """Generator to iterate over all queries in the ground truth data."""
        for _dataset_name, _dataset_dict in ground_truth_data.items():
            for _graph_name, _query_list in _dataset_dict.items():
                for _query_dict in _query_list:
                    yield _dataset_name, _graph_name, _query_dict

    # Iterate over all queries, categorizing them and updating the JSON-compatible structure
    for dataset_name, graph_name, query_dict in tqdm(list(iter_queries()), unit='query'):
        # Skip queries that already have a category
        if len(ground_truth_categories_json[dataset_name][graph_name]) > query_dict['id']:
            continue

        # Categorize the query and store category names
        query_categories = get_query_categories(query_dict['sparql_partial_uri'], categories)
        ground_truth_categories_json[dataset_name][graph_name].append([c.name for c in query_categories])

    # Save the categorized queries to the output file
    with CATEGORIES_DATA.open('w', encoding='utf8') as f:
        json.dump(ground_truth_categories_json, f, indent=2)

    # Convert JSON-stored category names back into QueryCategory objects
    ground_truth_categories = {
        dataset_name: {
            graph_name: [
                [categories_dict[category_name] for category_name in query_categories_names]
                for query_categories_names in query_list
            ]
            for graph_name, query_list in dataset_dict.items()
        }
        for dataset_name, dataset_dict in ground_truth_categories_json.items()
    }

    return ground_truth_categories


def get_categorized_score_dict(
        results_dict: dict[str, RunResults],
        categories: list[QueryCategory],
        metric: Callable[[list], float],
        **filters,
) -> dict:
    """Compute a nested dictionary of scores for different prompt types, models, and query categories."""
    return {
        prompt_type: {
            model_code: {
                category.name: round(metric(
                    model_run_results.retrieve(prompts=prompt_type, query_categories=category, **filters)
                ), ROUND_SCORES_DIGITS)
                for category in categories
            }
            for model_code, model_run_results in results_dict.items()
        }
        for prompt_type in PROMPT_TYPES
    }


def plot_category_bars(
    prompt_type: str,
    score_name: str, 
    categories_count: dict, 
    model_scores_dict: dict,
    baseline_dict: dict,
    bars_max_width: float = 0.80,
):
    """Plots a grouped bar chart with multiple models' scores across different query categories and a baseline as a
    horizontal stepped line."""
    # Ensure all models have a corresponding color
    for llm_code in model_scores_dict:
        if llm_code not in LLMS_COLORS_DICT:
            raise ValueError(f"Missing color for model {llm_code}.")

    # Labels for the x-axis (query categories)
    categories = list(categories_count.keys())
    counts = list(categories_count.values())
    x_labels = [f'{cat}\n(#{count})' for cat, count in zip(categories, counts)]

    # Scores for each model
    scores = {
        llm_code: [llm_dict[cat] for cat in categories]
        for llm_code, llm_dict in model_scores_dict.items()
    }

    # Baseline scores
    assert len(baseline_dict) == 1, 'Baseline dictionary must contain exactly one key.'
    baseline_name = next(iter(baseline_dict.keys()))
    baseline_scores = [baseline_dict[baseline_name][cat] for cat in categories]

    x = np.arange(len(categories))  # X-axis locations
    width = bars_max_width / len(LLMS_ORDER)  # Width of the bars
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for each model
    for i, (model, model_scores) in enumerate(scores.items()):
        ax.bar(
            x + (i - (len(scores) - 1) / 2) * width,
            model_scores,
            width,
            label=model,
            color=LLMS_COLORS_DICT[model]
        )

    # Plot baseline as a horizontal line varying by category
    ax.plot(
        x, baseline_scores, marker='o', linestyle='-', color=BASELINE_COLOR, label=baseline_name, drawstyle='steps-mid'
    )

    # --- Formatting ---
    ax.set_title(f'{prompt_type.upper()} {score_name} scores on different types of queries')
    ax.set_xlabel('Query categories')
    ax.set_axisbelow(True)
    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    # Y-axis
    ax.set_ylabel(score_name)
    max_y = 1.00
    y_step_major, y_step_minor = 0.10, 0.02
    ax.set_yticks(np.arange(0, max_y+y_step_major/2, y_step_major), minor=False)
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8, alpha=0.6)  # Primary lines
    ax.set_yticks(np.arange(0, max_y+y_step_minor/2, y_step_minor), minor=True)
    ax.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.5, alpha=0.3)  # Secondary lighter lines
    plt.ylim(0, max_y)
    # Legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.3)  # Adjust the bottom margin to fit the legend

    # Save the plot
    file_path = SCORES_DIR.joinpath(f'plot_{prompt_type}_{score_name.lower().replace(" ", "_")}.png')
    plt.savefig(str(file_path), dpi=300, bbox_inches='tight')  # Saves as PNG with high resolution


def round_max_score(model_scores_dict: dict, ceil: bool, step: float):
    """Finds the maximum score in a nested dictionary and rounds it up to the nearest multiple of `step`."""
    math_round_fun = math.ceil if ceil else math.floor
    max_score = max(score for score_dict in model_scores_dict.values() for score in score_dict.values())
    return round(math_round_fun(max_score / step) * step, 2)


def store_mostly_incorrect_queries(llms_run_results: dict, llms_run_dict: dict):
    """Identifies queries where most models provided incorrect answers and stores a sample of these queries."""
    # Select response based on correctness from "basic" or "detailed" prompts
    llm_queries_dict = {
        llm_code: [
            ('basic', basic) if (basic and basic.is_correct) else ('detailed', detailed)
            for basic, detailed in zip(
                llm_run_result.retrieve(
                    prompts='basic', most_voted=True, default_iteration=0
                ),
                llm_run_result.retrieve(
                    prompts='detailed', most_voted=True, default_iteration=0
                )
            )
        ]
        for llm_code, llm_run_result in llms_run_results.items()
    }

    llm_model_queries = {
        model_name: [
            q
            for dataset_name, graph_dict in dataset_dict.items()
            for graph_name, graph_queries in graph_dict.items()
            for q in graph_queries
        ]
        for model_name, dataset_dict in llms_run_dict.items()
    }

    # Load ground truth data
    with GROUND_TRUTH_PATH.open('r', encoding='utf8') as f:
        ground_truth_data = json.load(f)

    # Identify queries with high incorrect response rates
    wrong_queries_list = []
    i = 0
    for dataset_name, dataset_dict in ground_truth_data.items():
        for graph_name, graph_queries in dataset_dict.items():
            for q in graph_queries:
                wrong_model_queries = [
                    llm_run_results[i]
                    for llm_code, llm_run_results in llm_queries_dict.items()
                    if llm_run_results[i][1] and not llm_run_results[i][1].is_correct
                ]

                if len(wrong_model_queries) >= INCORRECT_ANSWER_THRESHOLD:
                    llm_wrong_queries = {}
                    for llm_code, llm_run_results in llm_queries_dict.items():
                        prompt = llm_run_results[i][0]
                        evaluation_data = llm_model_queries[llm_code][i]['evaluation'][prompt]
                        selected_iteration = evaluation_data['most_voted_id'] if evaluation_data['most_voted_id'] else 0
                        query = simplify_query(evaluation_data['repetitions'][selected_iteration]['generated_sparql'])
                        llm_wrong_queries[llm_code] = query
                    wrong_queries_list.append({
                        'ground_truth': simplify_query(q['sparql_partial_uri']),
                        **llm_wrong_queries
                    })
                i += 1

    # Write sampled incorrect queries to file
    with WRONG_QUERIES_PATH.open('w', encoding='utf8') as f:
        sample = (
            random.sample(wrong_queries_list, MAX_WRONG_QUERY_SAMPLES)
            if len(wrong_queries_list) > MAX_WRONG_QUERY_SAMPLES
            else wrong_queries_list
        )
        json.dump(sample, f, indent=2)


def simplify_query(q: str) -> str:
    """Removes prefix declaration and convert full URIs in partial URIs."""
    lines = q.split('\n')
    prefixes = [line.split(': ')[1].strip()[1:-1] for line in lines if line.startswith('PREFIX')]
    new_q = '\n'.join(lines[len(prefixes):])
    for prefix in prefixes:
        full_uri_pattern = rf'<{prefix}([^\<\>]+)>'
        new_q = re.sub(full_uri_pattern, r':\1', new_q)
    return new_q


if __name__ == '__main__':
    main()
