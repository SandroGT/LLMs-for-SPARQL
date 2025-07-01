import owlready2 as owl

from llms.abstract import AgentLLM
from agents.utils import parse_llm_answer, get_classes_str, get_properties_str

SYSTEM_TEMPLATE = '''\
You are a powerful NLP and SPARQL Interpreter tool.

Your task is to reason step-by-step through the mapping between the user's question and the query patterns needed\
 to retrieve valuable information for the proper answer. After reasoning, you will produce a valid SPARQL query that\
 retrieves the answer.

# Classes
{classes}

# Relations
{relations}

# Attributes
{attributes}

# Instructions

Follow these reasoning steps explicitly before generating the query:

### Step 1: Analyze the question
Identify:
- The main entity or type(s) involved (relevant classes).
- The focus of the question (what information is being asked).
- Any filters or conditions (e.g., values, constraints, comparisons).
- Any aggregation, ordering, or limits.
- The expected shape of the answer (single value, list, boolean, etc.).

### Step 2: Plan the query
Decide:
- Which variables will be used and what they represent.
- Which classes, properties (object or data), and filters are needed.
- How to join them in the WHERE clause.
- How to structure the SELECT clause (e.g., with DISTINCT, COUNT, ORDER BY, etc.).

### Step 3: Generate the query
Write the final SPARQL query using `:` as prefix for all ontology entities, **without defining any PREFIX clause**.

# Final Output
Respond with:
1. Your reasoning steps (Step 1 and Step 2).
2. The final SPARQL query **enclosed in a fenced triple backtick (```) code block**, and nothing else inside the block.\

You only produce one SPARQL query as the very final step of your answer.\
'''

USER_TEMPLATE = '''\
# Question
{question}
'''


class SparqlGenerationCoT(AgentLLM):
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
