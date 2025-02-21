import re

import owlready2 as owl


class ParsingException(Exception):
    pass


SKIP_TAGS = [
    '</think>'  # DeepSeek-r1
]


def parse_llm_answer(
        answer: str,
        base_iri: str,
        classes: set[owl.ThingClass],
        relations: set[owl.ObjectProperty],
        attributes: set[owl.DataProperty],
) -> str | ParsingException:
    """Parses and formats the SPARQL query answer."""
    try:
        # Extract text after tags
        for tag in SKIP_TAGS:
            if tag in answer:
                answer = re.search(rf'(?<={tag}).*', answer, re.DOTALL).group(0)

        # Remove any PREFIX declarations from the LLM, as they are not to be trusted
        answer = re.sub(r'''PREFIX .*\n''', '', answer, re.IGNORECASE)

        # Replace shortened entity names (e.g., :ClassName) with their full URIs for classes, relations, and attributes
        for e in classes | relations | attributes:
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


def get_name(entity: owl.EntityClass) -> str:
    """Extracts and formats the name of an ontology entity."""
    # Get the base IRI of the ontology entity's namespace
    base_iri = entity.namespace.ontology.base_iri

    # Remove the base IRI from the full IRI and replace optionals '#' with '_'
    name = entity.iri.replace(base_iri, '').replace('#', '_')

    # Handle owl.Thing
    if name == 'Thing':
        name = 'Any class'

    return name


def get_classes_str(classes: set[owl.ThingClass]) -> str:
    """Converts a set of ontology classes into a formatted string."""
    return stringify_list([f'- {get_name(c)}' for c in classes], element_wrap='', element_separator='\n')


def get_properties_str(
        properties: set[owl.Property]
) -> str:
    """Converts a set of ontology properties into a formatted string."""
    return stringify_list([
        f'- {get_name(p)} : {dr_info}'
        for p in properties
        if (dr_info := _get_domain_range_info_str(p)) is not None
    ], element_wrap='', element_separator='\n')


def _get_domain_range_info_str(entity: owl.PropertyClass) -> str | None:
    """Returns a formatted string representing the domain and range of an ontology property."""
    _domain, _range = entity.domain, entity.range
    if not _domain and not _range:
        return None

    # Stringify domain
    domain_names = _expand_constraint(_domain)
    domain_str = stringify_list(domain_names, element_wrap='', element_separator='|', list_wrap=('[', ']'))

    # Stringify range
    range_names = _expand_constraint(_range)
    range_str = stringify_list(range_names, element_wrap='', element_separator='|', list_wrap=('[', ']'))

    domain_range_str = f'{domain_str} -> {range_str}'
    return domain_range_str


def _expand_constraint(constraint: list | owl.Or | owl.ThingClass | type) -> list:
    if isinstance(constraint, list):
        return [ec for c in constraint for ec in _expand_constraint(c)]
    elif isinstance(constraint, owl.Or):
        return [ec for c in constraint.Classes for ec in _expand_constraint(c)]
    elif isinstance(constraint, owl.ThingClass):
        return [get_name(constraint)]
    elif isinstance(constraint, type):
        return [constraint.__name__]
    else:
        raise RuntimeError("Unprocessable constraint.")


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
