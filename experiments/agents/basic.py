import owlready2 as owl

from llms.abstract import AgentLLM
from agents.utils import parse_llm_answer, get_name, stringify_list

SYSTEM_TEMPLATE = '''\
Given an input question and an ontology create a syntactically correct SPARQL query.

# Output
Use the prefix `:` for all ontology entities, but don't define any `PREFIX` in the query.
Your answer must be the SPARQL query only.\
'''

USER_TEMPLATE = '''\
# Ontology
{{'classes': {classes}, 'object_properties': {object_properties}, 'data_properties': {data_properties}}}

# Question
{question}\

Your answer must be the SPARQL query only.\
'''


class SparqlGenerationBasic(AgentLLM):
    global SYSTEM_TEMPLATE, USER_TEMPLATE
    _SYSTEM_TEMPLATE: str = SYSTEM_TEMPLATE
    _USER_TEMPLATE: str = USER_TEMPLATE

    def _format_user(
            self,
            question: str,
            onto_classes: set[owl.ThingClass],
            onto_relations: set[owl.ObjectProperty],
            onto_attributes: set[owl.DataProperty],
            **_
    ) -> str:
        classes_str = stringify_list(
            [get_name(c) for c in onto_classes], element_wrap="'", element_separator=', ', list_wrap=('[', ']')
        )
        relations_str = stringify_list(
            [get_name(r) for r in onto_relations], element_wrap="'", element_separator=', ', list_wrap=('[', ']')
        )
        attributes_str = stringify_list(
            [get_name(a) for a in onto_attributes], element_wrap="'", element_separator=', ', list_wrap=('[', ']')
        )
        return self._USER_TEMPLATE.format(
            question=question,
            classes=classes_str,
            object_properties=relations_str,
            data_properties=attributes_str,
        )

    def _parse_answer(
            self,
            answer: str,
            base_iri: str,
            onto_classes: set[owl.ThingClass],
            onto_relations: set[owl.ObjectProperty],
            onto_attributes: set[owl.DataProperty],
            **_
    ) -> str | Exception:
        return parse_llm_answer(answer, base_iri, onto_classes, onto_relations, onto_attributes)
