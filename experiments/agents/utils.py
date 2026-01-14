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
            answer = re.sub(rf'(?<=\s):{get_name(e)}(?=[\s;.])', f'<{e.iri}>', answer)
            # answer = re.sub(rf'(?<=[\s\n{{]):{get_name(e)}(?=[\s\n;.}}])', f'<{e.iri}>', answer)
            # Replace occurrences of ClassName inside angle brackets with the full URI
            answer = re.sub(rf'(?<=<){get_name(e)}(?=>)', f'{e.iri}', answer)

        # Extract the SPARQL query from the formatted answer
        sparql_query = extract_sparql_query(answer)

        # Add prefixes
        sparql_query = f'PREFIX : <{base_iri}>\n{sparql_query}'  # This is not strictly necessary
        if 'xsd:' in sparql_query:
            sparql_query = f'PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n{sparql_query}'
        if 'rdf:' in sparql_query:
            sparql_query = f'PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n{sparql_query}'
        if 'rdfs:' in sparql_query:
            sparql_query = f'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n{sparql_query}'

        return sparql_query

    except Exception as e:
        return ParsingException(str(e))


def is_query_line(line: str) -> bool:
    """Determines whether a given line is likely part of a SPARQL query."""
    line = line.strip()

    # Empty lines are valid SPARQL
    if not line:
        return True

    # Common SPARQL keywords
    sparql_keywords = {
        'SELECT', 'ASK', 'CONSTRUCT', 'DESCRIBE', 'WHERE', 'FILTER', 'OPTIONAL', 'UNION', 'MINUS', 'ORDER',
        'GROUP', 'BY', 'GROUP BY', 'LIMIT', 'OFFSET', 'VALUES', 'PREFIX', 'BASE', 'AS', 'A'
    }

    # SPARQL syntax patterns
    variable_pattern = re.compile(r'\s?\?\w+\s?')
    uri_pattern = re.compile(r'\s?<[^>]+>\s?')
    prefix_pattern = re.compile(r'\s?:[A-Za-z]\w*\s?')
    brace_pattern = re.compile(r'\s?[{}()\[\]]\s?')

    # Create a regex pattern that excludes SPARQL keywords
    keywords_pattern = r'\b(?:' + '|'.join(map(re.escape, sparql_keywords)) + r')\b'

    # Sentence-like pattern (normal words that are not SPARQL keywords)
    sentence_like = re.compile(r'\s(?!' + keywords_pattern + r')\w+\s', re.IGNORECASE)

    # Check if the first word is a SPARQL keyword
    starts_with_keyword = any(line.strip().upper().startswith(w) for w in sparql_keywords)

    # Check for SPARQL structural patterns
    contains_sparql_syntax = (
        bool(variable_pattern.search(line)) or
        bool(uri_pattern.search(line)) or
        bool(prefix_pattern.search(line)) or
        bool(brace_pattern.search(line))
    )
    contains_normal_wordings = len(sentence_like.findall(line)) > 2

    # If a variable or URI is detected, but the line looks like a full sentence, ignore it
    if contains_sparql_syntax and contains_normal_wordings:
        return False

    return starts_with_keyword or contains_sparql_syntax


def extract_sparql_query(text: str) -> str:
    """Extracts the SPARQL query from an LLM response, handling code blocks and inline queries."""

    # 1. Check for a code block (``` ... ```)
    extracted_block = False
    for code_delimiter in ['```', '"""']:
        code_block_pattern = re.search(
            rf'{code_delimiter}(?:\w+)?\s*([\s\S]+?)\s*{code_delimiter}', text, re.IGNORECASE
        )
        if code_block_pattern:
            extracted_block = True
            text = code_block_pattern.group(1).strip()
    if extracted_block:
        return text

    # 2. If no code block, extract based on query-like lines
    query_lines = []
    in_query = False

    for line in text.split("\n"):
        # Detect the start of a query
        if is_query_line(line.strip()):
            query_lines.append(line)
            in_query = True
        # Stop if an explanation follows (heuristic: full sentence without SPARQL elements)
        elif in_query:
            break

    return '\n'.join(query_lines).strip()


def get_name(entity: owl.EntityClass) -> str:
    """Extracts and formats the name of an ontology entity."""
    if entity.namespace.ontology.base_iri == 'http://dbpedia.org/':
        clean_name = re.sub(r'\W', '_', entity.name)
        clean_name = re.sub(r'_+', '_', clean_name)
        clean_name = re.sub(r'^_|_$', '', clean_name)
        return clean_name
    else:
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
        f'- {get_name(p)} : {dr_info}' if (dr_info := _get_domain_range_info_str(p)) is not None else f'- {get_name(p)}'
        for p in properties
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
