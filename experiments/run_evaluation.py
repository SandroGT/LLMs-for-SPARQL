import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from logger import LOGGER
from evaluation.comparison import get_most_voted_result
from evaluation.metrics import (avg_accuracy, avg_generation_time, avg_parsing_efficiency, avg_syntax_correctness, avg_determinism)
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

# Support data paths
SUPPORT_DIR = EVALUATION_DIR.joinpath('support_data')
GROUPINGS_DATA = SUPPORT_DIR.joinpath('groupings.json')
CATEGORIES_DATA = SUPPORT_DIR.joinpath('categories.json')

# Specific evaluation results
EVALUATION_PATH = EVALUATION_DIR.joinpath('evaluation.json')
GROUND_TRUTH_PATH = RESULTS_DIR.joinpath('ground_truth.json')
BASELINE_PATH = RESULTS_DIR.joinpath('sgpt.json')

# === Constants ===
GROUPING_MAX_QUERY_TIME = 120
ROUND_SCORES_DIGITS = 4

# Model and prompt settings
LLMS_ORDER = ['gpt-3.5-turbo', 'gpt-4o-mini', 'llama-3.3-70b']
PROMPT_TYPES = ['basic', 'detailed']
ENSEMBLE_NAME = 'ensemble'

# Color mappings for visualization
COLORS_DICT = {
    'gpt-3.5-turbo': '#FFB3C6',
    'gpt-4o-mini': '#FF8FAB',
    'llama-3.1-8b': '#DDE7C7',
    'llama-3.3-70b': '#BFD8BD',
    'codestral-22b': '#B3DEE2'
}
# SEE https://coolors.co/efb8b1-c6e8d6-b2e0c8-bddcea-a6d0e3-f4ddb5-f5d1c7
colors_1 = ['#FFB3C6', '#FF8FAB', '#DDE7C7', '#BFD8BD', '#B3DEE2']
colors_2 = ['#F7A399', '#F38375', '#DDE7C7', '#BFD8BD', '#B3DEE2']


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
    llms_run_results = load_llms_run(query_categories)

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
    scores_accuracy_dict = get_score_dict(llms_run_results, categories, avg_accuracy, most_voted=True)

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
    with SCORES_DIR.joinpath('accuracy.json').open('w', encoding='utf8') as f:
        json.dump(scores_accuracy_dict | {'sgpt': sgpt_accuracy_dict}, f, indent=2)

    # Compute and store additional evaluation metrics
    scores_data = [
        ('generation_time.json', avg_generation_time, {'most_voted': False}),
        ('parsing_efficiency.json', avg_parsing_efficiency, {'most_voted': False}),
        ('syntax_correctness.json', avg_syntax_correctness, {'most_voted': False}),
        ('determinism.json', avg_determinism, {'return_iterations': True}),
    ]
    for filename, metric, filters in scores_data:
        result_dict = get_score_dict(llms_run_results, categories, metric, **filters)
        with SCORES_DIR.joinpath(filename).open('w', encoding='utf8') as f:
            json.dump(result_dict, f, indent=2)


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


def load_llms_run(query_categories: dict) -> dict[str, RunResults]:
    """Load LLMs run data from a JSON file and map it to a `RunResults` data structure."""
    LOGGER.info("Loading and processing zero-shot LLM runs.")
    llm_data = dict()

    # Load stored LLM-generated queries, skipping ground truth and other reference files
    for file in [RESULTS_DIR.joinpath(f'{name}.json') for name in LLMS_ORDER]:
        if file.name not in {'ground_truth.json', 'sgpt.json', 'llama-3.1-8b.json'}:
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
    for llm_code, llm_dict in llm_data.items():
        new_groupings_needed = llm_code not in groupings_data
        if new_groupings_needed:
            groupings_data[llm_code] = dict()

        for dataset_name, dataset_dict in llm_dict.items():
            if new_groupings_needed:
                groupings_data[llm_code][dataset_name] = dict()

            for graph_name, query_list in dataset_dict.items():
                if new_groupings_needed:
                    groupings_data[llm_code][dataset_name][graph_name] = list()
                    LOGGER.info(f'Processing groupings for {llm_code}-{dataset_name}/{graph_name}')
                    queries_pbar = tqdm(enumerate(query_list), total=len(query_list), unit='query')
                else:
                    queries_pbar = enumerate(query_list)

                graph_path = DATASET_DIR.joinpath(dataset_name, TEST_GRAPH_DIR, f'{graph_name}.rdf')

                for query_id, query_dict in queries_pbar:
                    if new_groupings_needed:
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
        with GROUPINGS_DATA.open('w', encoding='utf8') as f:
            json.dump(groupings_data, f, indent=2)

    # Final save to ensure all groupings are written
    with GROUPINGS_DATA.open('w', encoding='utf8') as f:
        json.dump(groupings_data, f, indent=2)

    # Return the structured data as a RunResults object
    return {
        llm_code: RunResults(llm_dict)
        for llm_code, llm_dict in llm_data.items()
    }


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


def get_score_dict(
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
    scores_dict: dict, 
    baseline_dict: dict,
    bars_max_width: float = 0.80,
):
    """Plots a grouped bar chart with multiple models' scores across different query categories and a baseline as a
    horizontal stepped line."""
    # Ensure all models have a corresponding color
    for llm_code in scores_dict:
        if llm_code not in COLORS_DICT:
            raise ValueError(f"Missing color for model {llm_code}.")

    # Labels for the x-axis (query categories)
    categories = list(categories_count.keys())
    counts = list(categories_count.values())
    x_labels = [f'{cat}\n(#{count})' for cat, count in zip(categories, counts)]

    # Scores for each model
    scores = {
        llm_code: [llm_dict[cat] for cat in categories]
        for llm_code, llm_dict in scores_dict.items()
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
        ax.bar(x + (i-1) * width, model_scores, width, label=model, color=COLORS_DICT[model])

    # Plot baseline as a horizontal line varying by category
    ax.plot(x, baseline_scores, marker='o', linestyle='-', color='black', label=baseline_name, drawstyle='steps-mid')

    # Formatting
    ax.set_xlabel('Query categories')
    ax.set_ylabel(score_name)
    ax.set_title(f'{prompt_type.upper()} {score_name} scores on different types of queries')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    # Save the plot
    file_path = SCORES_DIR.joinpath(f'plot_{prompt_type}_{score_name.lower().replace(" ", "_")}.png')
    plt.savefig(str(file_path), dpi=300, bbox_inches='tight')  # Saves as PNG with high resolution


if __name__ == '__main__':
    main()
