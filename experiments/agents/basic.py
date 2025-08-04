import owlready2 as owl

from llms.abstract import AgentLLM
from agents.utils import parse_llm_answer, get_name, stringify_list

SYSTEM_TEMPLATE = '''\
Given an input question and an ontology create a syntactically correct SPARQL query.

# Ontology
{{'classes': {classes}, 'object_properties': {object_properties}, 'data_properties': {data_properties}}}

# Output
Use the prefix `:` for all ontology entities, but don't define any `PREFIX` in the query.
Your answer must be the SPARQL query only.\
'''

USER_TEMPLATE = '''\
# Question
{question}
'''


class SparqlGenerationBasic(AgentLLM):
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
        classes_str = stringify_list(
            [get_name(c) for c in classes], element_wrap="'", element_separator=', ', list_wrap=('[', ']')
        )
        relations_str = stringify_list(
            [get_name(r) for r in relations], element_wrap="'", element_separator=', ', list_wrap=('[', ']')
        )
        attributes_str = stringify_list(
            [get_name(a) for a in attributes], element_wrap="'", element_separator=', ', list_wrap=('[', ']')
        )
        system_msg = self._SYSTEM_TEMPLATE.format(
            classes=classes_str,
            object_properties=relations_str,
            data_properties=attributes_str,
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
