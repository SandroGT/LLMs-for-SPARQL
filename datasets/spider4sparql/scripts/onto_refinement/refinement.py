"""
Functions for supervising and refining teh dataset ontologies.

Allow manual renaming of ontology entities while ensuring consistency in the RDF graph and SPARQL queries.
"""

import os
import re
from typing import Iterator

import owlready2 as owl

from logger import LOGGER
from uri_handler import name_from_uri, label_from_stem

ONTO_PRINT_TEMPLATE = """\
# Ontology - {name}

### Classes
{classes}

### Object properties
{object_properties}

### Data properties
{data_properties}
"""


def remove_id_suffixes(graph: owl.Ontology, uri_map: dict[str, str]) -> None:
    """Automatically renames ontology object properties to remove "id" suffixes."""
    # Some object properties are incorrectly named with "id" as part of the relation, which can confuse LLMs
    remove_id_regex = r'([\-_])?id$'
    for relation in graph.object_properties():
        if re.search(remove_id_regex, relation.name, flags=re.IGNORECASE):
            backup_uri = relation.iri
            backup_name = relation.name
            relation.name = re.sub(remove_id_regex, '', relation.name, flags=re.IGNORECASE)
            relation.label = label_from_stem(relation.name)  # Update label based on new name
            uri_map[backup_uri] = relation.iri  # Update mapping with new URI
            LOGGER.info(
                f"Updating '{type(relation).__name__}' entity from '{backup_name}' to '{relation.name}'."
            )
    # Ensure that the URIs remain unique after renaming
    assert len(uri_map.keys()) == len(set(uri_map.values()))


def interactive_ontology_supervision(graph: owl.Ontology, uri_map: dict[str, str]) -> None:
    """Interactively supervises and renames ontology entities in the given RDF graph.

    Entities (classes, object properties, and data properties) are indexed and printed for review. The user can input
    new names, which are then applied to the graph while maintaining the correct URI mappings.

    The process continues interactively until the user exits."""
    # Create a lookup dictionary for entities by index
    entities_lookup_dict = {idx: e for idx, e in enumerate(get_onto_entities(graph))}

    # Start interactive supervision
    print_onto(graph.name, entities_lookup_dict)  # Display current ontology
    entity_index, new_entity_name = get_input(uri_map)  # Get user input
    while entity_index is not None:
        try:
            update_graph(entity_index, new_entity_name, entities_lookup_dict, uri_map)  # Apply name update
        finally:
            print_onto(graph.name, entities_lookup_dict)  # Reprint updated ontology
            entity_index, new_entity_name = get_input(uri_map)  # Get next input


def print_onto(graph_name: str, entities_lookup_dict: dict[int, owl.EntityClass]) -> None:
    """Displays the ontology structure in a properly formatted way."""
    # Format ontology entities by type
    classes_str = '\n'.join([
        format_entity(i, e) for i, e in entities_lookup_dict.items() if isinstance(e, owl.ThingClass)
    ])
    obj_prop_str = '\n'.join([
        format_entity(i, e) for i, e in entities_lookup_dict.items() if isinstance(e, owl.ObjectPropertyClass)
    ])
    data_prop_str = '\n'.join([
        format_entity(i, e) for i, e in entities_lookup_dict.items() if isinstance(e, owl.DataPropertyClass)
    ])

    # Fill the ontology print template with formatted entity lists
    compiled_onto_template = ONTO_PRINT_TEMPLATE.format(
        name=graph_name,
        classes=classes_str,
        object_properties=obj_prop_str,
        data_properties=data_prop_str,
    )

    cls()  # Clear the console before printing
    print(compiled_onto_template)  # Display the formatted ontology


def format_entity(i: int, e: owl.EntityClass) -> str:
    """Formats an ontology entity into a readable string for display."""
    i_str = f'[{i:3}]'
    e_str = f'{e.name}'

    if isinstance(e, owl.PropertyClass):
        prop_str = _get_domain_range_info_str(e)
        return f'{i_str} {e_str} | ({prop_str})'
    else:
        return f'{i_str} {e_str}'


def cls() -> None:
    """Clears the terminal screen based on the operating system."""
    if os.name == 'nt':
        os.system('cls')
    elif os.name == 'posix':
        os.system('clear')
    else:
        raise OSError(f"Unsupported OS: {os.name}")


def get_input(uri_map: dict[str, str]) -> tuple[int, str]:
    """Prompts the user to input an entity index and a new name, returning them as a tuple."""
    names_in_use = {name_from_uri(new_uri) for new_uri in uri_map.values()}
    retry, entity_index, entity_new_name = True, None, None
    while retry:
        # Normally, `input(<prompt>)` should print to stdout, but the prompt was ending up mixed with the logging
        #  outputs in stderr. That apparently stupid print is there for that reason.
        print("Insert '<index>;<new name>' [only 'Enter' to finish]: ", end='')
        answer = input().strip()

        # Process answer
        if answer:
            input_index, input_new_name = None, None
            try:
                # Expecting input format: "<index>;<new name>"
                input_index, input_new_name = answer.split(';')
                input_index = int(input_index)
                assert input_new_name
            except Exception:
                print("Invalid format.")
            if input_index and not (0 <= input_index < len(names_in_use)):
                print(f"Index out of range.")
            elif input_new_name and (input_new_name not in names_in_use):
                entity_index, entity_new_name = input_index, input_new_name
                retry = False  # Valid input, exit loop
            elif input_new_name:
                print(f"Name '{input_new_name}' already in use.")
        else:
            retry = False  # Empty input means finishing

    return entity_index, entity_new_name


def update_graph(
        entity_index: int,
        entity_new_name: str,
        entities_lookup_dict: dict[int, owl.EntityClass],
        uri_map: dict[str, str]
) -> None:
    """Updates the name and label of an ontology entity, ensuring the URI map remains consistent."""
    entity_to_update = entities_lookup_dict[entity_index]
    LOGGER.info(
        f"Updating '{type(entity_to_update).__name__}' entity from '{entity_to_update.name}' to '{entity_new_name}'."
    )

    # Preserve the original URI before renaming
    backup_uri = entity_to_update.iri

    # Update the entity's name and label
    entity_to_update.name = entity_new_name
    entity_to_update.label = label_from_stem(entity_to_update.name)

    # Ensure the URI mapping reflects the change
    uri_map[backup_uri] = entity_to_update.iri


def _get_domain_range_info_str(entity: owl.PropertyClass) -> str:
    """Returns a formatted string representing the domain and range of an ontology property."""
    # Assumptions based on preprocessing:
    # - Each property has exactly one domain and one range axiom.
    # - The domain and range are expressed as an `Or` condition (alternative classes, not conjunctions).
    assert len(entity.domain) == 1 and len(entity.range) == 1  # Ensure single domain and range
    domain, range = entity.domain[0], entity.range[0]  # Extract single domain and range
    assert isinstance(domain, owl.Or) and isinstance(range, owl.Or)  # Ensure they are owl.Or

    # Extract and returns class names from domain and range
    domain = [getattr(c, 'name') for c in domain.Classes]
    range = [getattr(c, 'name', c.__name__) for c in range.Classes]  # c.__name__ is for data property raw values
    return f'{domain} -> {range}'


def update_query(query: str, uri_map: dict[str, str], partial_uri: bool = False) -> str:
    """Updates URIs in a SPARQL query according to the given mapping. If `partial_uri` is True, replaces prefixed names
    (e.g., `:oldName`) within the query. Otherwise, replaces full URIs (e.g., `<http://example.com/oldName>`)."""
    new_query = query
    for old_full_uri, new_full_uri in uri_map.items():
        if old_full_uri == new_full_uri:
            continue  # Skip unchanged URIs

        if partial_uri:
            old_regex = rf':{name_from_uri(old_full_uri)}(?=\s)'
            new_str = rf':{name_from_uri(new_full_uri)}'
        else:
            old_regex = rf'<{old_full_uri}>(?=\s)'
            new_str = rf'<{new_full_uri}>'

        new_query = re.sub(old_regex, new_str, new_query)

    return new_query


def get_onto_entities(graph: owl.Ontology) -> Iterator[owl.EntityClass]:
    """Yields all ontology entities (classes, object properties, and data properties) from the given RDF graph."""
    yield from graph.classes()
    yield from graph.object_properties()
    yield from graph.data_properties()
