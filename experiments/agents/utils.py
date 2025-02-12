import re

import owlready2 as owl


class ParsingException(Exception):
    pass


def parse_llm_answer(
        answer: str,
        base_iri: str,
        onto_classes: set[owl.ThingClass],
        onto_relations: set[owl.ObjectProperty],
        onto_attributes: set[owl.DataProperty],
) -> str | ParsingException:
    """Parses and formats the SPARQL query answer."""
    try:
        # Remove any PREFIX declarations from the LLM, as they are not to be trusted
        answer = re.sub(r'''PREFIX .*\n''', '', answer, re.IGNORECASE)

        # Replace shortened entity names (e.g., :ClassName) with their full URIs for classes, relations, and attributes
        for e in onto_classes | onto_relations | onto_attributes:
            # Replace occurrences of :ClassName with <full_uri> in the text
            answer = re.sub(f'(?<=\\s):{get_name(e)}(?=[\\s;.])', f'<{e.iri}>', answer)
            # Replace occurrences of ClassName inside angle brackets with the full URI
            answer = re.sub(f'(?<=<){get_name(e)}(?=>)', f'{e.iri}', answer)

        # Extract the SPARQL query from the formatted answer
        sparql_query = extract_sparql_query(answer)

        # Add prefixes
        sparql_query = f'PREFIX : <{base_iri}>\n{sparql_query}'
        if 'xsd:' in sparql_query:
            sparql_query = f'PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n{sparql_query}'
        if 'rdfs:' in sparql_query:
            sparql_query = f'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n{sparql_query}'

        return sparql_query

    except Exception as e:
        return ParsingException(str(e))


def is_query_line(line: str) -> bool:
    """Determines whether a given line is likely part of a SPARQL query."""
    # Pattern to match variables in SPARQL, e.g., ?var
    variable_pattern = re.compile(r'(\b)?\?\w+\b')

    # Pattern to match SPARQL keywords, which are typically uppercase (e.g., SELECT, WHERE)
    keyword_pattern = re.compile(r'\b([A-Z]{2,}[A-Z\s]*)\b')

    # Pattern to detect curly braces, commonly used for grouping in SPARQL
    curly_brace_pattern = re.compile(r'[{}]')

    # Pattern to detect opening or closing group delimiters (curly braces or parentheses)
    group_opening_and_closing = re.compile(r'([{(]\s*$)|(^\s*[)}])')

    # Pattern to detect URIs, which are enclosed in angle brackets (e.g., <http://example.org>)
    uri_pattern = re.compile(r'<[^>]+>')

    # Check if the line matches any of the defined SPARQL-related patterns
    return (
        bool(variable_pattern.search(line)) or
        bool(keyword_pattern.search(line)) or
        bool(curly_brace_pattern.search(line)) or
        bool(group_opening_and_closing.search(line)) or
        bool(uri_pattern.search(line))
    )


def extract_sparql_query(text: str) -> str:
    """Extracts the SPARQL query from a given text."""
    # Split the input text into individual lines
    lines = text.splitlines()

    # Identify the indices of lines that are part of the SPARQL query
    query_ids = [i for i, line in enumerate(lines) if is_query_line(line)]

    if query_ids:
        # Get the first and last line indices of the SPARQL query
        first_query_line_id = query_ids[0]
        last_query_line_id = query_ids[-1] + 1  # +1 to include the last query line

        # Join the identified lines and return the reconstructed query
        return '\n'.join(lines[first_query_line_id:last_query_line_id]).strip()
    else:
        # Return an empty string if no SPARQL query lines are found
        return ''


def get_name(onto_entity: owl.EntityClass) -> str:
    """Extracts and formats the name of an ontology entity."""
    # Get the base IRI of the ontology entity's namespace
    base_iri = onto_entity.namespace.ontology.base_iri

    # Remove the base IRI from the full IRI and replace optionals '#' with '_'
    return onto_entity.iri.replace(base_iri, '').replace('#', '_')


def get_classes_str(onto_classes: set[owl.ThingClass]) -> str:
    """Converts a set of ontology classes into a formatted string."""
    # Convert the ontology class set into a list of formatted names and join them with semicolons
    return stringify_list([get_name(c) for c in onto_classes], element_wrap='', element_separator='; ')


def get_relations_str(
        onto_classes: set[owl.ThingClass],
        onto_relations: dict[owl.ObjectProperty, dict[str, set[owl.ThingClass]]]
) -> str:
    """Converts ontology relations into a formatted string."""
    relations_str_list = list()

    # Iterate through each relation and its associated domain and range classes
    for r, r_dict in onto_relations.items():
        # Get domain class names that exist in the provided onto_classes set
        domain_class_names = [get_name(c) for c in r_dict['domain'] if c in onto_classes]
        domain_str = stringify_list(domain_class_names, element_wrap='', element_separator=', ')

        # Get range class names that exist in the provided onto_classes set
        range_class_names = [get_name(c) for c in r_dict['range'] if c in onto_classes]
        range_str = stringify_list(range_class_names, element_wrap='', element_separator=', ')

        # If both domain and range are present, add the relation to the list
        if domain_class_names and range_class_names:
            relations_str_list.append(f'([{domain_str}], {get_name(r)}, [{range_str}])')

    # Return the formatted string or 'No interesting relations' if no relations were found
    if relations_str_list:
        return stringify_list(relations_str_list, element_wrap='', element_separator='; ')
    else:
        return 'No interesting relations'


def get_attributes_str(
        onto_classes: set[owl.ThingClass],
        onto_attributes: dict[owl.DataProperty, dict[str, set[owl.ThingClass]]]
) -> str:
    """Converts ontology attributes into a formatted string."""
    attributes_str_list = list()

    # Iterate through each attribute and its associated domain classes and range types
    for a, a_dict in onto_attributes.items():
        # Get domain class names that exist in the provided onto_classes set
        domain_class_names = [get_name(c) for c in a_dict['domain'] if c in onto_classes]
        domain_str = stringify_list(domain_class_names, element_wrap='', element_separator=', ')

        # Get range type names
        range_type_names = [t.__name__ for t in a_dict['range']]
        range_str = stringify_list(range_type_names, element_wrap='', element_separator=', ')

        # If both domain and range are present, add the attribute to the list
        if domain_class_names and range_type_names:
            attributes_str_list.append(f'([{domain_str}], {get_name(a)}, [{range_str}])')

    # Return the formatted string or 'No interesting attributes' if no attributes were found
    if attributes_str_list:
        return stringify_list(attributes_str_list, element_wrap='', element_separator='; ')
    else:
        return 'No interesting attributes'


def stringify_list(
        elements_list: list,
        element_wrap: str = '"',
        element_separator: str = ', ',
        list_wrap: tuple | None = None
) -> str:
    """Converts a list of elements into a formatted string."""
    # Wrap each element in the list with the specified wrap character
    str_list = element_separator.join([f'{element_wrap}{e}{element_wrap}' for e in elements_list])

    # If list_wrap is provided, wrap the entire string with the start and end strings
    if list_wrap:
        assert isinstance(list_wrap, tuple) and len(list_wrap) == 2
        assert isinstance(list_wrap[0], str) and isinstance(list_wrap[1], str)
        str_list = list_wrap[0] + str_list + list_wrap[1]

    return str_list
