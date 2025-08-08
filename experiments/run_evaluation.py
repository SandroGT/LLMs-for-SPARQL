from http.client import RemoteDisconnected
import json
import math
from pathlib import Path
import random
import re
import string
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from dbpedia import run_sparql_query
from evaluation.comparison import get_most_voted_result
from evaluation.metrics import (avg_accuracy, avg_generation_time, avg_syntax_correctness, avg_determinism)
from evaluation.query_types import QueryCategory, get_query_categories
from evaluation.results import RunResults
from jena import JenaQuery
from logger import LOGGER
from timeout import set_timeout, TimeoutException

plt.rcParams.update({'font.size': 8})


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

PROMPT_TYPES = ['basic', 'detailed', 'cot']
ENSEMBLE_NAME = 'ensemble'
DATASETS = ['spider4sparql', 'bestiary', 'lcquad']
DATASET_NAMES = {'spider4sparql': 'Spider4SPARQL', 'bestiary': 'Bestiary', 'lcquad': 'LC-QuAD'}
INCORRECT_ANSWER_THRESHOLD = int(round(1.00 * len(LLMS_ORDER)))
MAX_WRONG_QUERY_SAMPLES = 30

# Color mappings for visualization
# SEE Colors for GPT, Llama, Phi and Cohere: https://coolors.co/ffbe8f-ffa375-ff855c-ddab88-d79575-ffd37a-c2f391-88f273
# SEE Colors for Mistral, Qwen and DeepSeek: https://coolors.co/b9f9e9-83fce8-97d4fc-7eb4fc-c6c0ec-bdaae9
LLMS_COLORS_DICT = {
    'gpt-3.5-turbo': '#FFBE8F',
    'gpt-4o-mini': '#FFA375',
    'gpt-4o': '#FF855C',
    'llama-3.1-8b': '#DDAB88',
    'llama-3.3-70b': '#D79575',
    'phi-4-14b': '#FFD37A',
    'c4ai-command-r-7b': '#C2F391',
    'c4ai-command-r-32b': '#88F273',
    'codestral-v0.1-22b': '#B9F9E9',
    'mistral-small-24b': '#83FCE8',
    'qwen-2.5-32b': '#97D4FC',
    'qwen-2.5-coder-32b': '#7EB4FC',
    'deepseek-v2-coder-16b': '#C6C0EC',
    'deepseek-r1-qwen-32b': '#BDAAE9',
}
BASELINE_COLOR = '#808080'
DEVIATION_COLOR = '#9D0208'

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
    category_all = [c for c in categories if c.name == 'all'][0]
    query_categories = categorize_queries(categories)

    # Load model outputs for evaluation
    sgpt_run_results = load_sgpt_run(query_categories)
    llms_run_dict, llms_run_results = load_llms_run(query_categories)

    # Compute baseline accuracy for SGPT
    sgpt_accuracy_dict = {
        'sgpt': {
            category.name: round(avg_accuracy(sgpt_run_results.retrieve(query_categories=category)), ROUND_SCORES_DIGITS)
            for category in categories
        }
    }

    # Compute accuracy variance across iterations of BASIC and DETAILED prompts for all datasets
    for prompt_type in ['basic', 'detailed']:
        variance_dict = dict()
        for llm_code, llm_run_result in llms_run_results.items():
            llm_dict = dict()
            variance_dict[llm_code] = llm_dict
            for dataset in DATASETS:
                results = llm_run_result.retrieve(
                    prompts=prompt_type, datasets=dataset, query_categories=category_all, return_iterations=True
                )
                accuracies_dict = {'best': list(), 'worst': list(), 'average': list()}

                for query_result in results:
                    correctness = [rep is not None and rep.is_correct for rep in query_result.repetitions_data]
                    accuracies_dict['best'].append(int(True in correctness))
                    accuracies_dict['worst'].append(int(False not in correctness))
                    accuracies_dict['average'].append(sum([int(c) for c in correctness]) / len(correctness))

                llm_dict[dataset] = {k: sum(l) / len(l) for k, l in accuracies_dict.items()}

        plot_repetition_bars(prompt_type, 'accuracy', variance_dict)

    # Compute and store accuracy table
    filename, metric, filters = ('accuracy.csv', avg_accuracy, {'most_voted': True})
    result_dict = {
        dataset: get_categorized_score_dict(llms_run_results, categories, metric, datasets=dataset, **filters)
        for dataset in DATASETS
    }
    for dataset in DATASETS:
        ensemble_accuracy_dict = dict()
        for llm_code, llm_run_result in llms_run_results.items():
            llm_dict = dict()
            ensemble_accuracy_dict[llm_code] = llm_dict
            for category in categories:
                basic_results, detailed_results, cot_results = [
                    llm_run_result.retrieve(
                        prompts=p, datasets=dataset, query_categories=category, most_voted=True
                    )
                    for p in ['basic', 'detailed', 'cot']
                ]
                ensemble_results = list()
                for prompt_results in zip(basic_results, detailed_results, cot_results):
                    correctness = [res is not None and res.is_correct for res in prompt_results]
                    ensemble_results.append(int(True in correctness))

                ensemble_acc = sum(ensemble_results) / len(ensemble_results) if len(ensemble_results) > 0 else 1
                llm_dict[category.name] = ensemble_acc

        result_dict[dataset][ENSEMBLE_NAME] = ensemble_accuracy_dict

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

    # Plot accuracy ENSEMBLE scores for Spider4SPARQL
    dataset = 'spider4sparql'
    score_name = 'accuracy'
    categories_count = {
        category.name: len(
            llms_run_results[LLMS_ORDER[0]].retrieve(
                prompts='basic', query_categories=category, datasets=dataset
            )
        )
        for category in categories
    }
    plot_category_bars(
        ENSEMBLE_NAME,
        score_name,
        dataset,
        categories_count,
        result_dict[dataset][ENSEMBLE_NAME],
        sgpt_accuracy_dict
    )

    # Compute and store additional evaluation metrics tables
    scores_data = [
        ('generation_time.csv', avg_generation_time, {'most_voted': False}),
        ('syntax_correctness.csv', avg_syntax_correctness, {'most_voted': False}),
        ('determinism.csv', avg_determinism, {'return_iterations': True}),
    ]
    for filename, metric, filters in scores_data:
        table_data = dict()
        for dataset_name in DATASETS:
            result_dict = get_categorized_score_dict(
                llms_run_results, categories, metric, datasets=dataset_name, **filters
            )
            table_data |= {
                f'{dataset_name}-{prompt_type}': {
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

    # Create a mapping from dataset name to proper query function
    local_jena_engine = JenaQuery()
    query_functions_dict = {
        'bestiary': local_jena_engine.run_query,
        'lcquad': run_sparql_query,
        'spider4sparql': local_jena_engine.run_query,
    }

    # Iterate over the LLMs run data to complete it
    LOGGER.info(f'Processing groupings')
    for llm_code, llm_dict in llm_data.items():
        if llm_code not in groupings_data:
            llm_groupings = dict()
            groupings_data[llm_code] = llm_groupings
        else:
            llm_groupings = groupings_data[llm_code]
            assert isinstance(llm_groupings, dict)

        for dataset_name, dataset_dict in llm_dict.items():
            query_fun = query_functions_dict[dataset_name]

            if dataset_name not in llm_groupings:
                dataset_groupings = dict()
                llm_groupings[dataset_name] = dataset_groupings
            else:
                dataset_groupings = llm_groupings[dataset_name]
                assert isinstance(dataset_groupings, dict)

            for graph_name, query_list in dataset_dict.items():
                if graph_name not in dataset_groupings:
                    graph_groupings = list()
                    dataset_groupings[graph_name] = graph_groupings
                else:
                    graph_groupings = dataset_groupings[graph_name]
                    assert isinstance(graph_groupings, list)

                # LOGGER.info(f'Processing groupings for {llm_code}-{dataset_name}/{graph_name}')
                queries_pbar = enumerate(query_list)  # tqdm(enumerate(query_list), total=len(query_list), unit='query')
                graph_path = DATASET_DIR.joinpath(dataset_name, TEST_GRAPH_DIR, f'{graph_name}.rdf')
                last_id = len(graph_groupings)-1
                for query_id, query_dict in queries_pbar:
                    if query_id > last_id:
                        query_groupings = dict()
                        graph_groupings.append(query_groupings)
                    else:
                        query_groupings = graph_groupings[query_id]
                        assert isinstance(query_groupings, dict)

                    for prompt_type, results_list in query_dict['evaluation'].items():
                        if prompt_type not in query_groupings:
                            # Execute SPARQL queries unless they previously failed (marked by execution_error)
                            query_results = list()

                            for res in results_list:
                                r = None
                                if res['execution_error'] is None:
                                    delays = [5, 15, 60]
                                    for attempt in range(len(delays) + 1):
                                        try:
                                            with set_timeout(GROUPING_MAX_QUERY_TIME):
                                                r = query_fun(query=res['generated_sparql'], graph_path=graph_path)
                                            break
                                        except TimeoutException:
                                            break
                                        except Exception as e:
                                            if attempt < len(delays):
                                                time.sleep(delays[attempt])
                                            else:
                                                raise e
                                query_results.append(r)

                            # Compute groupings based on query execution results
                            groups_index_sets, index_most_voted, _ = get_most_voted_result(query_results)
                            query_groupings[prompt_type] = {
                                'groups': [list(g) for g in groups_index_sets],
                                'most_voted_id': index_most_voted
                            }

                    # Update query dictionary with groupings
                    for prompt_type, results_list in query_dict['evaluation'].items():
                        query_dict['evaluation'][prompt_type] = {
                            'groups': query_groupings[prompt_type]['groups'],
                            'most_voted_id': query_groupings[prompt_type]['most_voted_id'],
                            'repetitions': results_list
                        }
                    query_dict['categories'] = query_categories[dataset_name][graph_name][query_dict['id']]

            # Periodic saving to avoid data loss in case of crashes
            with GROUPINGS_DATA.open('w', encoding='utf8') as f:
                json.dump(groupings_data, f, indent=2)

    # Final save to ensure all groupings are written
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
        query_categories = get_query_categories(query_dict['sparql_complete_uri'], categories)
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


def plot_repetition_bars(
        prompt_type: str,
        score_name: str,
        variance_dict: dict,  # model -> dataset -> 'best'/'worst'/'average' -> accuracy
        bars_max_width: float = 0.90,
):
    """
    Plots a grouped bar chart showing variance in accuracy across repetitions of prompts (e.g., BASIC/DETAILED).

    Args:
        prompt_type: Type of prompt used (e.g., "basic", "detailed").
        score_name: Name of the score being plotted (e.g., "accuracy").
        variance_dict: Dictionary with scores in format model -> dataset -> target -> accuracy.
        bars_max_width: Maximum total width allocated to all model bars at one x-tick.
    """
    # Ensure all models have a defined color
    for llm_code in variance_dict:
        if llm_code not in LLMS_COLORS_DICT:
            raise ValueError(f"Missing color for model {llm_code}.")

    datasets = list(next(iter(variance_dict.values())).keys())
    x_labels = [DATASET_NAMES[d] for d in datasets]
    x = np.arange(len(datasets))

    # Compute scores and error bars
    scores = dict()
    errors_lower = dict()
    errors_upper = dict()

    for llm_code, llm_scores in variance_dict.items():
        scores[llm_code] = []
        errors_lower[llm_code] = []
        errors_upper[llm_code] = []

        for dataset in datasets:
            best = llm_scores[dataset]['best']
            worst = llm_scores[dataset]['worst']
            average = llm_scores[dataset]['average']

            lower = max(average - worst, 0)
            upper = max(best - average, 0)

            scores[llm_code].append(average)
            errors_lower[llm_code].append(lower)
            errors_upper[llm_code].append(upper)

    # Bar width
    width = bars_max_width / len(variance_dict)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for each model
    for i, (model, model_scores) in enumerate(scores.items()):
        x_offset = x + (i - (len(scores) - 1) / 2) * width
        letter = string.ascii_letters[i].upper()

        asymmetric_error = [errors_lower[model], errors_upper[model]]

        ax.bar(
            x_offset,
            model_scores,
            width,
            yerr=asymmetric_error,
            capsize=2,
            error_kw={
                'ecolor': DEVIATION_COLOR,
                'elinewidth': 0.5,
                'capthick': 1
            },
            label=f"{letter}) {model}",
            color=LLMS_COLORS_DICT[model]
        )

        for xi in x_offset:
            ax.text(xi, -0.01, letter, ha='center', va='top', fontsize=5, fontweight='bold', clip_on=False)

    # --- Formatting ---
    title = f'{prompt_type.upper()} {score_name} scores variance across repetitions'
    ax.set_title(title)
    ax.set_xlabel('Datasets')
    ax.set_axisbelow(True)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis='x', which='major', pad=12, length=0)
    ax.set_xlim(left=min(x) - (2 - bars_max_width) / 2, right=max(x) + (2 - bars_max_width) / 2)

    max_y = 1.05
    y_step_major, y_step_minor = 0.10, 0.02
    ax.set_yticks(np.arange(0, max_y + y_step_major / 2, y_step_major), minor=False)
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8, alpha=0.6)
    ax.set_yticks(np.arange(0, max_y + y_step_minor / 2, y_step_minor), minor=True)
    ax.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.ylim(0, max_y)

    ax.legend(loc='upper center', bbox_to_anchor=(0.50, -0.18), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.20)

    # Save the plot
    file_path = SCORES_DIR.joinpath(f'plot_{prompt_type}_{score_name.lower().replace(" ", "_")}_variance.pdf')
    plt.savefig(str(file_path), format='pdf', bbox_inches='tight')


def plot_category_bars(
    prompt_type: str,
    score_name: str,
    plot_dataset: str,
    categories_count: dict,
    model_scores_dict: dict,
    baseline_dict: dict,
    bars_max_width: float = 0.90,
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
        x_offset = x + (i - (len(scores) - 1) / 2) * width
        letter = string.ascii_letters[i].upper()
        ax.bar(
            x_offset,
            model_scores,
            width,
            label=f"{letter}) {model}",
            color=LLMS_COLORS_DICT[model]
        )

        for xi in x_offset:
            ax.text(xi, -0.01, letter, ha='center', va='top', fontsize=5, fontweight='bold', clip_on=False)

    # Plot baseline as a horizontal line varying by category
    baseline_coordinates = (list(), list())
    for i, base_score in enumerate(baseline_scores):
        x1 = x[i] - 0.50
        x2 = x[i] + 0.50
        y = base_score
        baseline_coordinates[0].extend([x1, x2])
        baseline_coordinates[1].extend([y, y])
    ax.plot(
        baseline_coordinates[0],
        baseline_coordinates[1],
        marker='.',
        linestyle='-',
        color=BASELINE_COLOR,
        label=baseline_name
    )

    # --- Formatting ---
    title = f'{prompt_type.upper()} {score_name} scores on different types of queries'
    if plot_dataset is not None:
        plot_dataset = DATASET_NAMES[plot_dataset]
        title += f' from {plot_dataset}'
    ax.set_title(title)
    ax.set_xlabel('Query categories')
    ax.set_axisbelow(True)
    # X-axis
    ax.set_xticks(x, )
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis='x', which='major', pad=12, length=0)
    ax.set_xlim(left=min(x)-(2-bars_max_width)/2, right=max(x)+(2-bars_max_width)/2)
    max_y = 1.00
    y_step_major, y_step_minor = 0.10, 0.02
    ax.set_yticks(np.arange(0, max_y+y_step_major/2, y_step_major), minor=False)
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8, alpha=0.6)  # Primary lines
    ax.set_yticks(np.arange(0, max_y+y_step_minor/2, y_step_minor), minor=True)
    ax.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.5, alpha=0.3)  # Secondary lighter lines
    plt.ylim(0, max_y)
    # Legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.50, -0.18), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.20)  # Adjust the bottom margin to fit the legend

    # Save the plot
    file_path = SCORES_DIR.joinpath(f'plot_{prompt_type}_{score_name.lower().replace(" ", "_")}.pdf')
    plt.savefig(str(file_path), format='pdf', bbox_inches='tight')


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
                        'dataset': dataset_name,
                        'natural_language': q['nl_question'],
                        'ground_truth': simplify_query(q['sparql_complete_uri']),
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
