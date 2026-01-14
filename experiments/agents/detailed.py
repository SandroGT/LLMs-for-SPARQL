import owlready2 as owl

from llms.abstract import AgentLLM
from agents.utils import parse_llm_answer, get_classes_str, get_properties_str

SYSTEM_TEMPLATE = '''\
You are a powerful NLP and SPARQL Interpreter tool.

Your final goal is to find the information needed to answer a user question. So, your task is to analyze natural\
 language questions and construct SPARQL queries that provide valuable information for the proper answer.

# Classes
{classes}

# Relations
{relations}

# Attributes
{attributes}

# Instructions

### 1. Check the input
Carefully check the classes, relations and attributes that the user show you: choose all those that are necessary to\
 represent the query patterns of the answer.

### 2. Identify key elements
Determine the key components of the question and the answer to map them into SPARQL keywords:
- **Return information**: the focus of what is being asked? (Use `SELECT`, `ASK`);
- **Quantifiers**: how many? (Use `COUNT`);
- **Superlatives**: best, most, worst? (Use `ORDER BY` with optionally `ASC`, `DESC` and/or `LIMIT`, `MIN`, `MAX`, ...);
- **Constraints**: inequalities or conditions on specific values? (Use `FILTER` and `HAVING`);
- **Aggregations**: average, sum, etc.? (Use `AVG`, `SUM`, ...);
- **Grouping**: collect and aggregate data based on a common condition? (Use `GROUP BY`);
- **Exclusions**: with no, without any, where no one, ...? (Use `MINUS` or `FILTER NOT EXIST`);
- **Alternative options**: including alternative patterns "or"? (Use `UNION`).

### 3. Build the SPARQL query
Put together all the input data to construct a functioning SPARQL query, with proper variables. Pay attention to:
- Perfectly match the syntax and naming of classes, relations and attributes (including letter case).
- Use distinguished variables for different concepts.
- Use the prefix `:` for all graph classes and properties. Anyway, you dont have to define any PREFIX.
- Return exactly the information the query is asking for, not a variable more.

### 4. Final answer
Your answer must be the SPARQL query only.\
'''

USER_TEMPLATE = '''\
# Question
{question}
'''


class SparqlGenerationDetailed(AgentLLM):
    global SYSTEM_TEMPLATE, USER_TEMPLATE
    _SYSTEM_TEMPLATE: str = SYSTEM_TEMPLATE
    _USER_TEMPLATE: str = USER_TEMPLATE

    def _format_system(
            self,
            classes: set[owl.ThingClass],
            relations: set[owl.ObjectProperty],
            attributes: set[owl.DataProperty],
            **_
    ) -> str:
        assert classes and relations
        classes_str = get_classes_str(classes)
        relations_str = get_properties_str(relations)
        attributes_str = get_properties_str(attributes)
        system_msg = self._SYSTEM_TEMPLATE.format(
            classes=classes_str,
            relations=relations_str,
            attributes=attributes_str,
        )
        return system_msg

    def _format_user(
            self,
            question: str,
            **_
    ) -> str:
        user_msg = self._USER_TEMPLATE.format(
            question=question,
        )
        return user_msg

    def _parse_answer(
            self,
            answer: str,
            base_iri: str,
            classes: set[owl.ThingClass],
            relations: set[owl.ObjectProperty],
            attributes: set[owl.DataProperty],
            **_
    ) -> str | Exception:
        return parse_llm_answer(answer, base_iri, classes, relations, attributes)
