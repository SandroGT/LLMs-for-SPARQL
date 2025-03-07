from pathlib import Path
import re

import owlready2 as owl
import pandas as pd

from logger import LOGGER  # Assumes a custom logging module

# Define paths to input files
query_path = Path('..', 'processed', 'queries', 'dev', 'bestiary.csv')
graph_path = Path('..', 'processed', 'graph', 'dev', 'bestiary.rdf')

# Load SPARQL queries dataset
df_query = pd.read_csv(query_path)

# Load ontology from RDF file
onto = owl.get_ontology(str(graph_path)).load()


def main():
    """Identifies ontology elements used in SPARQL queries and logs unused elements."""

    # Extract concept names used in SPARQL queries (assumed prefixed with ':')
    used_concept_names = {
        match
        for _, (_, sparql, _) in df_query.iterrows()
        for match in re.findall(r'(?<=\s:)\w+(?=\s)', sparql)
    }

    # Collect ontology elements that are explicitly referenced in queries
    referenced_concepts = {
        'classes': {cls for cls in onto.classes() if cls.name in used_concept_names},
        'object properties': {prop for prop in onto.object_properties() if prop.name in used_concept_names},
        'data properties': {prop for prop in onto.data_properties() if prop.name in used_concept_names},
        'individuals': {ind for ind in onto.individuals() if ind.name in used_concept_names},
    }

    # Ensure all related classes are included:
    # - If an individual is used, add its classes.
    # - If an object/data property is used, add its domain and range classes.
    for individual in referenced_concepts['individuals']:
        referenced_concepts['classes'].update(individual.is_instance_of)

    for obj_prop in referenced_concepts['object properties']:
        referenced_concepts['classes'].update(obj_prop.domain)
        referenced_concepts['classes'].update(obj_prop.range)

    for data_prop in referenced_concepts['data properties']:
        referenced_concepts['classes'].update(data_prop.domain)

    # Log unused ontology elements
    log_unused_elements('classes', onto.classes(), referenced_concepts)
    log_unused_elements('object properties', onto.object_properties(), referenced_concepts)
    log_unused_elements('data properties', onto.data_properties(), referenced_concepts)


def log_unused_elements(element_type, all_elements, referenced_concepts):
    """Logs ontology elements of a given type that are not referenced in queries."""
    unused = {elem for elem in all_elements if elem not in referenced_concepts[element_type]}
    if unused:
        LOGGER.info(f'\nUnused {element_type}')
        for elem in unused:
            LOGGER.info(f' - {elem.name}')  # `.name` is clearer than slicing IRI


if __name__ == '__main__':
    main()
