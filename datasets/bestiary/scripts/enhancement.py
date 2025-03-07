import json
from pathlib import Path
import re
import shutil
import tempfile
import types

import owlready2 as owl
import pandas as pd
from tqdm import tqdm

from jena import JenaQuery
from logger import LOGGER

SCRIPT_PATH = Path(__file__).parent.resolve()
RAW_PATH = SCRIPT_PATH.joinpath('..', 'raw').resolve()
PROCESSED_PATH = SCRIPT_PATH.joinpath('..', 'processed').resolve()

RAW_QUERY_PATH = RAW_PATH.joinpath('queries.json').resolve()
RAW_GRAPH_PATH = RAW_PATH.joinpath('graph.rdf').resolve()
GRAPH_NAMESPACE = 'http://www.semanticweb.org/annab/ontologies/2022/3/ontology#'

QUERY_FOLDER = 'queries'
GRAPH_FOLDER = 'graph'
FOLD = 'dev'


def main():
    # Check if the raw data folder exists
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "The raw data folder does not exist. Please ensure the dataset is in the correct location."
        )

    # Create the output folders
    LOGGER.info(f"Creating output folders in '{PROCESSED_PATH}'.")
    if PROCESSED_PATH.exists():
        shutil.rmtree(PROCESSED_PATH)  # Remove existing output folder if it exists
    query_folder = PROCESSED_PATH.joinpath(QUERY_FOLDER, FOLD)
    query_folder.mkdir(parents=True)  # Create the query folder
    graph_folder = PROCESSED_PATH.joinpath(GRAPH_FOLDER, FOLD)
    graph_folder.mkdir(parents=True)  # Create the graph folder

    # Load and clean the ontology by removing annotation properties
    LOGGER.info(f"Loading ontology.")
    beast_ontology = load_clean_ontology()
    graph_store_path = PROCESSED_PATH.joinpath(GRAPH_FOLDER, FOLD, 'bestiary.rdf')
    beast_ontology.save(file=str(graph_store_path), format='rdfxml')

    # Load query data (questions and SPARQL queries)
    LOGGER.info(f"Loading and testing queries.")
    query_data, total, failed = load_and_test_queries(graph_store_path)
    LOGGER.info(f"Dropped {failed} queries due to execution failure. Kept {total-failed} queries.")

    # Extract beast categories from the queries
    categories_set, regex_count = get_beast_categories(query_data['sparql_complete_uri'])
    LOGGER.info(f"Found {regex_count} queries using 'FILTER regex' on entity URIs.")

    # Add beast type property and labels to the ontology
    add_beast_types_and_names(beast_ontology, categories_set)
    add_labels(beast_ontology)

    # Save the modified ontology to an RDF file
    graph_store_path = PROCESSED_PATH.joinpath(GRAPH_FOLDER, FOLD, 'bestiary.rdf')
    beast_ontology.save(file=str(graph_store_path), format='rdfxml')

    # Create query data-frame
    query_df = pd.DataFrame(query_data)
    query_store_path = PROCESSED_PATH.joinpath(QUERY_FOLDER, FOLD, 'bestiary.csv')
    query_df.to_csv(query_store_path, index=False)

    LOGGER.info(f"Stored graph and queries.")


def load_clean_ontology() -> owl.Ontology:
    """Loads an ontology from a raw RDF file, cleans up incorrect annotation properties, and replaces them with the
    correct data properties. The cleaned ontology is then loaded into an OWL world and returned."""
    # Read raw ontology RDF file
    with RAW_GRAPH_PATH.open('r', encoding='utf8') as f:
        rdf_text = f.read()

    # Pattern for an annotation property definition in RDF/XML format
    ann_p_definition_pattern = '<owl:AnnotationProperty rdf:about="{namespace}{name}"/>'

    # List of annotation properties to be corrected
    corrections = [
        {'ann_p': 'hasACValue', 'data_p': 'hasACvalue', 'new_data_p': 'hasACValue'},
        {'ann_p': 'hasCRValue', 'data_p': 'hasCRvalue', 'new_data_p': 'hasCRValue'},
        {'ann_p': 'hasHPvalue', 'data_p': 'hasHPValue', 'new_data_p': 'hasHPValue'},
    ]

    for correction in corrections:
        ann_p, data_p, new_data_p = correction['ann_p'], correction['data_p'], correction['new_data_p']
        definition = ann_p_definition_pattern.format(namespace=GRAPH_NAMESPACE, name=ann_p)

        # Remove the incorrect annotation property definition
        rdf_text = re.sub(rf'\n\s*{re.escape(definition)}\s*\n', '', rdf_text)

        # Replace incorrect annotation property usage with data property
        rdf_text = rdf_text.replace(ann_p, data_p)

        # Standardize to the correct data property name
        rdf_text = rdf_text.replace(data_p, new_data_p)

    # Remove wrongly defined positive integers (they have 0s assigned, throwing errors)
    rdf_text = rdf_text.replace(
        '"http://www.w3.org/2001/XMLSchema#positiveInteger"',
        '"http://www.w3.org/2001/XMLSchema#integer"'
    )



    # Use a temporary file to store the modified ontology and load it within the context
    with tempfile.NamedTemporaryFile(suffix=".owl", mode='w+', encoding='utf8', delete=True) as temp_file:
        temp_file.write(rdf_text)
        temp_file.flush()  # Ensure data is written before reading

        # Load the cleaned ontology into an OWL world
        world = owl.World()
        cleaned_ontology = world.get_ontology(temp_file.name).load()
        cleaned_ontology.name = PROCESSED_PATH.stem

    return cleaned_ontology


def load_and_test_queries(graph_path: Path) -> tuple[dict, int, int]:
    """Loads SPARQL queries and their corresponding natural language questions from the JSON file."""
    # Open and read the JSON file containing queries
    with RAW_QUERY_PATH.open('r', encoding='utf8') as f:
        query_json = json.load(f)

    # Instantiate query engine
    query_engine = JenaQuery()

    # Dictionary to store extracted data
    df_dict = {
        'nl_question': [],  # List of natural language questions
        'sparql_partial_uri': [],  # SPARQL queries with full URIs
        'sparql_complete_uri': []  # SPARQL queries with prefixed URIs
    }

    # Iterate through the list of questions in the JSON file
    total = len(query_json['questions'])
    failed = 0
    for question_dict in tqdm(query_json['questions']):
        # Extract the English-language question
        en_question = next(qd['string'] for qd in question_dict['question'] if qd['language'] == 'en')

        # Extract the original SPARQL query
        sparql_complete_query = question_dict['query']['sparql']

        # Add prefixes, when required
        if 'xsd:' in sparql_complete_query:
            sparql_complete_query = f'PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n{sparql_complete_query}'
        if 'rdfs:' in sparql_complete_query:
            sparql_complete_query = f'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n{sparql_complete_query}'

        # Replace full URIs with prefixed notation using GRAPH_NAMESPACE
        sparql_partial_uri = re.sub(rf'<{GRAPH_NAMESPACE}([^>]*)>', r':\1', sparql_complete_query)
        sparql_partial_uri = f'PREFIX : <{GRAPH_NAMESPACE}>\n{sparql_partial_uri}'

        try:
            query_engine.run_query(graph_path, sparql_partial_uri)
            runs = True
        except Exception as e:
            runs = False

        if runs:
            # Store the extracted data
            df_dict['nl_question'].append(en_question)
            df_dict['sparql_partial_uri'].append(sparql_partial_uri)
            df_dict['sparql_complete_uri'].append(sparql_complete_query)
        else:
            failed += 1

    return df_dict, total, failed


def get_beast_categories(sparql_queries: list[str]) -> tuple[set[str], int]:
    """Extracts beast-related categories from the list of SPARQL queries.

    Some SPARQL queries in the dataset use "FILTER regex" to filter URIs by category (e.g., "robots", "dragons").
    This is an inefficient way to model data in a Knowledge Graph (KG) but is a peculiarity of this dataset.
    The function identifies such categories and counts how many queries use this approach."""
    categories_set = set()  # Stores unique categories found in FILTER regex clauses
    regex_queries_count = 0  # Counts the number of queries using regex filters

    for sq in sparql_queries:
        # Match FILTER regex clauses that extract category names from URIs
        class_match = re.search(r'''FILTER regex\(.*,\s?\'(?P<class>.*)\'\s?,\s?\'i\'\s?\)''', sq)
        if class_match:
            categories_set.add(class_match.group('class'))
            regex_queries_count += 1

    return categories_set, regex_queries_count


def add_beast_types_and_names(onto: owl.Ontology, categories: set[str]) -> None:
    """Enhances the ontology by adding beast category and name properties to individuals."""
    with onto:
        # Define a new data property for categorizing beasts
        new_prop_name = 'beastCategoryName'
        beast_category_property = types.new_class(new_prop_name, (owl.DataProperty,))
        beast_category_property.domain = [owl.class_construct.Or([onto.Beast])]
        beast_category_property.range = str  # Category is represented as a string

        # Assign beast categories to individuals based on IRI
        for i in onto.individuals():
            if isinstance(i, onto.Beast):
                for c in categories:
                    if c in i.iri.lower():
                        getattr(i, new_prop_name).append(c)

        # Define a new data property for storing names
        new_prop_name = 'hasName'
        has_name_property = types.new_class(new_prop_name, (owl.DataProperty,))
        has_name_property.domain = [owl.class_construct.Or([owl.Thing])]
        has_name_property.range = str  # Name is represented as a string

        # Assign human-readable names to individuals
        for i in onto.individuals():
            if onto.Language in i.is_instance_of and i.iri.endswith('L'):
                iri = i.iri[:-1]
            else:
                iri = i.iri
            getattr(i, new_prop_name).append(get_label(iri, False))  # Lower name
            getattr(i, new_prop_name).append(get_label(iri, True))   # Capitalized name


def get_label(uri: str, capitalize: bool) -> str:
    """Generate a human-readable name from a URI stem."""
    uri_stem = uri[len(GRAPH_NAMESPACE):]
    name = uri_stem
    # Format name: handle camelCase and snake_case
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)  # Insert space in camelCase
    name = name.replace('_', ' ')  # Replace underscores with spaces
    if capitalize:
        # Typically for classes
        name = ' '.join(word.capitalize() for word in name.split())  # Capitalize each word
    else:
        # Typically for properties
        name = name.lower()
    return name


def add_labels(onto: owl.Ontology) -> None:
    """Assigns human-readable labels to ontology classes and properties."""
    # Assign capitalized labels to ontology classes
    for c in onto.classes():
        c.label = get_label(c.iri, True)

    # Assign lowercase labels to object properties
    for c in onto.object_properties():
        c.label = get_label(c.iri, False)

    # Assign lowercase labels to data properties
    for c in onto.data_properties():
        c.label = get_label(c.iri, False)


if __name__ == '__main__':
    main()
