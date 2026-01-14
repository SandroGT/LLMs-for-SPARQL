# LCQuAD - Ontology snapshot creation

This README describes the **ontology snapshot creation** step for the **LCQuAD** dataset based on **DBpedia**.  
This step generates small, **closed sub-ontologies** derived from the DBpedia knowledge graph to support
**zero-shot NL-to-SPARQL** generation with large language models (LLMs).

## Purpose

The LCQuAD dataset uses DBpedia, an open knowledge graph with a massive and unbounded schema.  
For LLMs operating in **zero-shot** mode, this is problematic:  
- The full ontology is **too large to include in the prompt**,  
- And if omitted, the model **lacks schema awareness** needed to produce valid SPARQL.

To address this, we generate **category-based subsets** of the DBpedia ontology:
- We parse each SPARQL query in LCQuAD to extract **classes** and **properties**.
- We resolve each class's `rdf:type` and retain only **schema.org types** as coarse categories.
- We group queries, classes, and properties by category.
- We split any category whose size exceeds a threshold into multiple parts (e.g., `person`, `person_2`, etc.)
- We keep only categories with a **manageable size** (3–10 classes/properties) to ensure they fit in the LLM's prompt.

Each category becomes a **self-contained, closed graph snapshot** that includes:
- A trimmed OWL ontology with just the required classes and properties.
- A list of the natural language questions and SPARQL queries associated with that subset.

## Filtering

To ensure meaningful evaluation, we keep **only the SPARQL queries that yield results** when executed against DBpedia.

This is essential for evaluation strategies based on comparing the query results: if the ground truth query returns no
 results, any query that also returns nothing would look "correct", even if it isn't semantically equivalent.  
Therefore, all queries that do **not return results** are discarded before any categorization or export step.

We also deduplicate queries across categories, so that each query is assigned to **at most one** subgraph snapshot.

## Execution

To start the creation process, ensure the `src` folder of this repository is in `PYTHONPATH`, then:

1. Download the LCQuAD test set from the official repository:  
   https://github.com/AskNowQA/LC-QuAD/blob/data/test-data.json

2. Place the raw `.json` file in the following directory:  
   `datasets/dbpedia/raw/`

3. Run the following command to generate the ontology snapshots and query subsets:

```
python lcquad_snapshot_generator.py
```

## Output folder structure

After running the script, the processed sub-ontologies and their corresponding queries will be saved in the following
 structure:

```
processed/
├── graph/
│   ├── <category ontology>.rdf
│   └── ...
└── queries/
    ├── <category queries>.csv
    └── ...
```

Each RDF file defines a **mini ontology** (classes + properties) for a given schema.org category.  
Each CSV file contains the **LCQuAD entries** that use those classes/properties and can be used to test LLM
performance in that subdomain.
