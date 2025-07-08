import json
from pathlib import Path
import re
import shutil
import types

import owlready2 as owl
import pandas as pd
from tqdm import tqdm

from dbpedia import run_sparql_query
from logger import LOGGER

# -------------------------------
# Configuration and Constants
# -------------------------------

DBPEDIA_BASE_URI = r'http://dbpedia.org/'
CLASS_RE_KEY = 'resource'
PROPERTY_RE_KEYS = ['property', 'ontology']
RESOURCE_RE_TEMPLATE = r'(?<=\<)http://dbpedia\.org/(?:{res_type})/[^>]*(?=\>)'

SCRIPT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR.parent
assert DATASET_DIR.name == 'lcquad', "Script must be in a 'lcquad/' subfolder."

# Input and cache paths
LCQUAD_RAW_TEST_PATH = DATASET_DIR.joinpath('raw', 'test-data.json')
assert LCQUAD_RAW_TEST_PATH.exists(), "LCQuAD test set not found."

LCQUAD_EXEC_CACHE = SCRIPT_DIR.joinpath('cache_exec_queries.json')
LCQUAD_CATEGORIES_CACHE = SCRIPT_DIR.joinpath('cache_categories.json')

# Output directories
OUTPUT_DIR = DATASET_DIR.joinpath('processed')
OUTPUT_ONTO_DIR = OUTPUT_DIR.joinpath('graph', 'dev')
OUTPUT_QUERIES_DIR = OUTPUT_DIR.joinpath('queries', 'dev')

# Ontology snapshot constraints
MIN_CATEGORY_COUNT = 5
MAX_CATEGORY_COUNT = 20


# -------------------------------
# Main Pipeline
# -------------------------------

def main():
    """
    Main execution pipeline:
    - Filters LCQuAD queries that return results
    - Groups remaining queries by schema.org type
    - Filters valid categories by class/property count
    - Deduplicates queries across categories
    - Exports ontology snapshots and query CSVs
    """
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_ONTO_DIR.mkdir(parents=True)
    OUTPUT_QUERIES_DIR.mkdir(parents=True)

    # Time-consuming (cached)
    lcquad_data = get_executable_lcquad()

    # Time-consuming (cached)
    query_categorization = get_query_categorization(lcquad_data)

    # Lightweight, run every time
    selected_categories = get_valid_categories(query_categorization)
    deduplicate_queries(query_categorization, selected_categories)
    validate_category_integrity(query_categorization, lcquad_data, selected_categories)
    create_outputs(lcquad_data, query_categorization, selected_categories)


# -------------------------------
# Data Preprocessing
# -------------------------------

def load_raw_lcquad() -> list[dict]:
    """
    Loads the original LCQuAD test set and returns it as a list of query dictionaries.

    Each entry in the list is a dictionary with the following fields:
    - "_id" (str): Unique string identifier for the question-query pair.
    - "corrected_question" (str): The natural language question, cleaned and human-readable.
    - "intermediary_question" (str): A semi-structured representation of the question, showing the key entities and predicates.
    - "sparql_query" (str): The full SPARQL query to be executed over DBpedia.
    - "sparql_template_id" (int): ID of the template used to generate this query from the underlying data.

    Returns:
        list[dict]: A list of structured question-query entries from the LCQuAD test set.
    """
    with LCQUAD_RAW_TEST_PATH.open() as f:
        return json.load(f)


def get_executable_lcquad() -> list[dict]:
    """
    Filters LCQuAD queries by checking if they yield non-empty results on DBpedia.
    Only SELECT queries are considered; ASK queries are kept by default.

    Returns:
        list of dicts: Each containing 'nl_question' and 'sparql_query'.
    """
    if LCQUAD_EXEC_CACHE.exists():
        LOGGER.info("Loading cached LCQuAD entries that yield results.")
        with LCQUAD_EXEC_CACHE.open('r', encoding='utf8') as f:
            return json.load(f)

    lcquad_data = load_raw_lcquad()
    LOGGER.info('Preprocessing LCQuAD: filtering queries that yield results.')

    filtered_data = []
    for i, query_dict in (pbar := tqdm(enumerate(lcquad_data, 1), total=len(lcquad_data))):
        query_str = query_dict['sparql_query'].strip()

        keep = True
        if not query_str.startswith('ASK'):
            assert query_str.startswith('SELECT'), f"Unexpected query type: {query_str[:10]}"
            results = run_sparql_query(query_str, get_bindings=True)
            keep = len(results) > 0

        if keep:
            cleaned_entry = {
                'nl_question': re.sub(r'\s+', ' ', query_dict['corrected_question']).strip(),
                'sparql_query': query_str
            }
            filtered_data.append(cleaned_entry)

        pbar.set_postfix({'keeping': f'{len(filtered_data)}/{i}'})

    LOGGER.info(f'Filtered out {len(lcquad_data) - len(filtered_data)} queries (no results).')
    LOGGER.info(f'Caching filtered LCQuAD queries.')
    with LCQUAD_EXEC_CACHE.open('w', encoding='utf8') as f:
        json.dump(filtered_data, f, indent=2)

    return filtered_data


# -------------------------------
# Categorization by Schema Type
# -------------------------------

def get_query_categorization(lcquad_data: list[dict]) -> dict:
    """
    Groups each query by its schema.org category using the rdf:type of involved DBpedia resources.
    If a category exceeds MAX_CATEGORY_COUNT in classes/properties, a new numbered category is created.

    Returns:
        dict: { category_uri or category_uri_2: { query_ids, classes, properties } }
    """
    if LCQUAD_CATEGORIES_CACHE.exists():
        LOGGER.info("Loading cached query categorization.")
        with LCQUAD_CATEGORIES_CACHE.open('r', encoding='utf-8') as f:
            return json.load(f)

    LOGGER.info("Grouping filtered queries by schema.org categories with overflow chunking.")
    categorization = {}
    category_counter = {}  # Track suffix per schema category

    for i, qd in tqdm(enumerate(lcquad_data), total=len(lcquad_data)):
        sparql_q = qd['sparql_query']
        q_classes = set(re.findall(RESOURCE_RE_TEMPLATE.format(res_type=CLASS_RE_KEY), sparql_q))
        q_props = set(re.findall(RESOURCE_RE_TEMPLATE.format(res_type='|'.join(PROPERTY_RE_KEYS)), sparql_q))

        for class_uri in q_classes:
            for type_uri in get_types(class_uri):
                # Track suffix index for this category
                base_uri = type_uri
                suffix = category_counter.get(base_uri, 1)

                while True:
                    key = base_uri if suffix == 1 else f"{base_uri}_{suffix}"
                    cat = categorization.setdefault(key, {'query_ids': [], 'classes': [], 'properties': []})

                    new_class_count = len(set(cat['classes']) | q_classes)
                    new_prop_count = len(set(cat['properties']) | q_props)

                    if new_class_count > MAX_CATEGORY_COUNT or new_prop_count > MAX_CATEGORY_COUNT:
                        suffix += 1
                        category_counter[base_uri] = suffix
                        continue
                    else:
                        # Safe to add
                        cat['query_ids'] = list(set(cat['query_ids']) | {i})
                        cat['classes'] = list(set(cat['classes']) | q_classes)
                        cat['properties'] = list(set(cat['properties']) | q_props)
                        break

    LOGGER.info("Caching query categorization by schema.org type.")
    with LCQUAD_CATEGORIES_CACHE.open('w', encoding='utf-8') as f:
        json.dump(categorization, f, indent=2)

    return categorization


def get_valid_categories(categorization: dict) -> set[str]:
    """
    Filters schema.org categories to include only those with a reasonable number
    of involved classes and properties.

    Returns:
        set: Selected schema.org type URIs
    """
    selected = set()

    for type_uri, info in categorization.items():
        assert len(info['classes']) <= MAX_CATEGORY_COUNT and len(info['properties']) <= MAX_CATEGORY_COUNT
        if (
            MIN_CATEGORY_COUNT <= len(info['classes']) and
            MIN_CATEGORY_COUNT <= len(info['properties'])
        ):
            selected.add(type_uri)

    LOGGER.info(f'Identified {len(selected)} valid DBpedia sub-ontology snapshots/graphs.')

    return selected


def deduplicate_queries(categorization: dict, selected_categories: set) -> None:
    """
    Deduplicates queries across selected categories.

    Keeps each query in the first category it appears in and removes it from the rest.
    If a category ends up with no queries, it is removed entirely.

    WARNING: This function modifies both `categorization` and `selected_categories` in place.
    """
    seen = set()
    removed = 0
    to_remove = set()

    for category_name in selected_categories:
        category_info = categorization[category_name]
        original_ids = category_info['query_ids']
        filtered_ids = [i for i in original_ids if i not in seen]
        removed += len(original_ids) - len(filtered_ids)

        if filtered_ids:
            category_info['query_ids'] = filtered_ids
            seen.update(filtered_ids)
        else:
            to_remove.add(category_name)

    for cat in to_remove:
        selected_categories.remove(cat)
        del categorization[cat]

    LOGGER.info(f'Deduplicated queries across categories: removed {removed} duplicates.')
    LOGGER.info(f'{len(seen)} unique queries remain across {len(selected_categories)} snapshots.')


def validate_category_integrity(categorization: dict, lcquad_data: list[dict], selected_categories: set) -> None:
    """
    Ensures that every class and property mentioned in the queries of each selected category
    is included in that category's recorded class/property lists.

    Raises:
        AssertionError if a mismatch is found.
    """
    for category_name in selected_categories:
        category_info = categorization[category_name]
        known_classes = set(category_info['classes'])
        known_properties = set(category_info['properties'])

        for query_id in category_info['query_ids']:
            sparql = lcquad_data[query_id]['sparql_query']
            query_classes = set(re.findall(RESOURCE_RE_TEMPLATE.format(res_type=CLASS_RE_KEY), sparql))
            query_properties = set(re.findall(RESOURCE_RE_TEMPLATE.format(res_type='|'.join(PROPERTY_RE_KEYS)), sparql))

            missing_classes = query_classes - known_classes
            missing_properties = query_properties - known_properties

            if missing_classes or missing_properties:
                raise AssertionError(
                    f"[{category_name}] Query {query_id} refers to missing resources:\n"
                    f"  Classes: {missing_classes}\n"
                    f"  Properties: {missing_properties}"
                )

    LOGGER.info("All queries are consistent with their category's ontology snapshot.")


# -------------------------------
# Output Creation
# -------------------------------

def create_outputs(lcquad_data: list[dict], categorization: dict, categories: set[str]):
    """
    Exports ontology snapshots (OWL) and related query CSVs for selected categories.
    """
    for type_uri in categories:
        info = categorization[type_uri]
        query_ids = info['query_ids']
        classes = info['classes']
        properties = info['properties']

        world = owl.World()
        onto = world.get_ontology(DBPEDIA_BASE_URI)

        with onto:
            for class_uri in classes:
                cls = types.new_class(class_uri, (owl.Thing,))
                name, label = process_uri(class_uri, capitalize=True)
                cls.iri = class_uri
                cls.label = label
                assert cls.name == name

            for prop_uri in properties:
                prop = types.new_class(prop_uri, (owl.ObjectProperty,))
                name, label = process_uri(prop_uri, capitalize=False)
                prop.iri = prop_uri
                prop.label = label
                assert prop.name == name

        assert len(list(onto.classes())) == len(classes)
        assert len(list(onto.properties())) == len(properties)

        # File naming
        category_name = type_uri.split('/')[-1]
        file_base = re.sub(r'(?<!^)(?<![A-Z])([A-Z])', r'_\1', category_name).lower()

        # Save OWL ontology
        onto_path = OUTPUT_ONTO_DIR.joinpath(f'{file_base}.rdf')
        onto.save(file=str(onto_path), format='rdfxml')

        # Save CSV queries
        rows = []
        for qid in query_ids:
            entry = lcquad_data[qid]
            rows.append({
                'nl_question': entry['nl_question'],
                'sparql_partial_uri': '',
                'sparql_complete_uri': entry['sparql_query']
            })
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_QUERIES_DIR.joinpath(f'{file_base}.csv'), index=False)


# -------------------------------
# Utility
# -------------------------------

def get_types(class_uri: str) -> list[str]:
    """
    Retrieves schema.org rdf:types for a given DBpedia resource.
    """
    query = rf"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT DISTINCT ?type
    WHERE {{
      <{class_uri}> rdf:type ?type .
    }}
    """
    results = run_sparql_query(query, get_bindings=True)
    return [v for r in results if 'schema.org' in (v := r['type']['value'])]


def process_uri(uri: str, capitalize: bool) -> tuple[str, str]:
    """
    Converts a URI to an internal OWL class/property name and a human-readable label.
    """
    name = uri.split('/')[-1]
    if '_' in name:
        label = name.replace('_', ' ')
    else:
        label = re.sub(r'(?<!^)(?<![A-Z])([A-Z])', r' \1', name).lower()
    label = ' '.join(w.lower().capitalize() if capitalize else w.lower() for w in label.split())
    return name, label


# -------------------------------
# Entrypoint
# -------------------------------

if __name__ == '__main__':
    main()
