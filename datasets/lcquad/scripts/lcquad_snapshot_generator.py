import json
import re
import shutil
import types
from pathlib import Path

import owlready2 as owl
import pandas as pd
from tqdm import tqdm

from dbpedia import run_sparql_query

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
LCQUAD_TEST_PATH = DATASET_DIR.joinpath('raw', 'test-data.json')
assert LCQUAD_TEST_PATH.exists(), "LCQuAD test set not found."

LCQUAD_CACHE_STORE_PATH = SCRIPT_DIR.joinpath('processing_cache.json')

# Output directories
PROCESSED_LCQUAD_DIR = DATASET_DIR.joinpath('processed')
PROCESSED_LCQUAD_ONTO_DIR = PROCESSED_LCQUAD_DIR.joinpath('graph', 'dev')
PROCESSED_LCQUAD_QUERIES_DIR = PROCESSED_LCQUAD_DIR.joinpath('queries', 'dev')

# Min/max elements per ontology snapshot
MIN_CATEGORY_COUNT = 5
MAX_CATEGORY_COUNT = 20

# -------------------------------
# Main Logic
# -------------------------------

def main():
    """
    Main pipeline:
    - Parses SPARQL queries from LCQuAD
    - Groups them by schema.org parent class
    - Filters categories with manageable ontology sizes
    - Exports OWL ontology snapshots and related queries
    """
    # Clean output folders
    if PROCESSED_LCQUAD_DIR.exists():
        shutil.rmtree(PROCESSED_LCQUAD_DIR)
    PROCESSED_LCQUAD_ONTO_DIR.mkdir(parents=True)
    PROCESSED_LCQUAD_QUERIES_DIR.mkdir(parents=True)

    # Create categories cache
    if not LCQUAD_CACHE_STORE_PATH.exists():
        try:
            lcquad_data, query_categorization = lcquad_data_build()
        except Exception as e:
            LCQUAD_CACHE_STORE_PATH.unlink(missing_ok=True)
            raise e
    else:
        lcquad_data, query_categorization = lcquad_data_load()

    tracked_categories = set()
    tracked_queries = list()

    # Filter based on class/property count
    for type_category, info in query_categorization.items():
        queries, classes, properties = [info[k] for k in ['query_ids', 'classes', 'properties']]
        if (
            MIN_CATEGORY_COUNT <= len(classes) <= MAX_CATEGORY_COUNT and
            MIN_CATEGORY_COUNT <= len(properties) <= MAX_CATEGORY_COUNT
        ):
            tracked_categories.add(type_category)
            tracked_queries.extend(queries)

    # Summary
    total_queries = len(tracked_queries)
    unique_queries = len(set(tracked_queries))
    redundant_queries = total_queries - unique_queries
    print(f'Identified {len(tracked_categories)} valid DBpedia sub-ontology snapshots.')
    print(f'Grouped {total_queries} queries, with {redundant_queries} repeated across categories'
          f' ({unique_queries} unique).')

    # Export OWL files and queries
    for type_category in tracked_categories:
        info = query_categorization[type_category]
        queries, classes, properties = [info[k] for k in ['query_ids', 'classes', 'properties']]

        world = owl.World()
        onto = world.get_ontology(DBPEDIA_BASE_URI)

        # Build ontology
        with onto:
            for class_uri in classes:
                onto_class = types.new_class(class_uri, (owl.Thing,))
                name, label = process_uri(class_uri, capitalize=True)
                onto_class.iri = class_uri
                onto_class.label = label
                assert onto_class.name == name

            for prop_uri in properties:
                onto_prop = types.new_class(prop_uri, (owl.ObjectProperty,))
                name, label = process_uri(prop_uri, capitalize=False)
                onto_prop.iri = prop_uri
                onto_prop.label = label
                assert onto_prop.name == name

        assert len(list(onto.classes())) == len(classes)
        assert len(list(onto.properties())) == len(properties)

        # Save ontology
        category_name = type_category.split('/')[-1]
        file_base = re.sub(r'(?<!^)([A-Z])', r'_\1', category_name).lower()
        onto_path = PROCESSED_LCQUAD_ONTO_DIR.joinpath(f'{file_base}.rdf')
        onto.save(file=str(onto_path), format='rdfxml')

        # Save related queries as CSV
        queries_path = PROCESSED_LCQUAD_QUERIES_DIR.joinpath(f'{file_base}.csv')
        rows = []
        for q_id in queries:
            entry = lcquad_data[q_id]
            nl_question = entry.get('corrected_question', '').replace('\n', ' ').strip()
            sparql_query = entry.get('sparql_query', '').replace('\n', ' ').strip()
            rows.append({
                'nl_question': nl_question,
                'sparql_partial_uri': '',
                'sparql_complete_uri': sparql_query
            })
        df = pd.DataFrame(rows)
        df.to_csv(queries_path, index=False)


# -------------------------------
# Helper Functions
# -------------------------------

def process_uri(uri: str, capitalize: bool) -> tuple[str, str]:
    """
    Converts a DBpedia URI into a valid OWL class/property name and a readable label.

    Args:
        uri (str): Full DBpedia URI
        capitalize (bool): Whether to capitalize words in the label

    Returns:
        tuple[str, str]: (internal_name, human-readable label)
    """
    name = uri.split('/')[-1]
    label = name.replace('_', ' ')
    label = ' '.join(w.lower().capitalize() if capitalize else w.lower() for w in label.split())
    return name, label


def lcquad_data_build() -> tuple[dict, dict]:
    """
    Processes the LCQuAD dataset and groups SPARQL queries by their schema.org parent class.

    For each query:
    - Extracts DBpedia resource classes and properties
    - Retrieves schema.org types via rdf:type
    - Groups queries, classes, and properties under each type

    Returns:
        tuple:
            - lcquad_data: Original LCQuAD dataset (list of dicts)
            - query_categorization: Mapping schema.org type to associated queries/classes/properties
    """
    with LCQUAD_TEST_PATH.open() as f:
        lcquad_data = json.load(f)

    categorization = dict()

    for i, qd in tqdm(enumerate(lcquad_data), total=len(lcquad_data)):
        sparql_q = qd['sparql_query']

        # Extract resources and properties
        q_classes = set(re.findall(RESOURCE_RE_TEMPLATE.format(res_type=CLASS_RE_KEY), sparql_q))
        q_props = set(re.findall(RESOURCE_RE_TEMPLATE.format(res_type='|'.join(PROPERTY_RE_KEYS)), sparql_q))

        for class_uri in q_classes:
            for type_uri in get_types(class_uri):
                if type_uri not in categorization:
                    categorization[type_uri] = {'query_ids': [], 'classes': [], 'properties': []}

                categorization[type_uri]['query_ids'] = list(set(categorization[type_uri]['query_ids']) | {i})
                categorization[type_uri]['classes'] = list(set(categorization[type_uri]['classes']) | q_classes)
                categorization[type_uri]['properties'] = list(set(categorization[type_uri]['properties']) | q_props)

        # Save after each query for safety
        with LCQUAD_CACHE_STORE_PATH.open('w', encoding='utf-8') as f:
            json.dump(categorization, f, indent=2)

    return lcquad_data, categorization


def get_types(class_uri: str) -> list[str]:
    """
    Retrieves rdf:types for a DBpedia class, filtered to include only schema.org types.

    Args:
        class_uri (str): Full URI of the class/resource (e.g. http://dbpedia.org/resource/Albert_Einstein)

    Returns:
        list[str]: A list of schema.org rdf:type URIs
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


def lcquad_data_load() -> tuple[dict, dict]:
    """
    Loads the original LCQuAD data and cached categorization (from processing_cache.json).

    Returns:
        tuple:
            - lcquad_data (list): List of original LCQuAD queries
            - query_categorization (dict): Grouping of queries/classes/properties by schema.org type
    """
    with LCQUAD_TEST_PATH.open() as f:
        lcquad_data = json.load(f)
    with LCQUAD_CACHE_STORE_PATH.open('r', encoding='utf-8') as f:
        query_categorization = json.load(f)
    return lcquad_data, query_categorization


if __name__ == '__main__':
    main()
