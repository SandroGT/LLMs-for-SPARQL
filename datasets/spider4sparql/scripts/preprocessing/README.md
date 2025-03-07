# Spider4SPARQL - Preprocessing

This README describes the preprocessing step for the **Spider4SPARQL** dataset, which is designed for the **natural
 language to SPARQL translation task**. The preprocessing script prepares the dataset for the next processing steps by
 cleaning and standardizing the data.

## Dataset overview

The [Spider4SPARQL dataset](https://github.com/ckosten/Spider4SPARQL/) includes:
- **Knowledge Graphs (KGs)**: materialized KGs in Turtle (`.ttl`) format, split into `dev` and `train` folds.
- **Ontologies**: explicit ontologies in `.owl` format for `dev` KGs, detailing class and property definitions.
- **Query Database**: a database (in `.csv` and `.json` formats) containing **(natural language question, SPARQL
   query)** pairs associated with each KG.

## Purpose

The preprocessing script addresses issues related to data quality, ontology consistency, and query validity.
 Specifically, it:

1. **Cleans up files**:
    - Removes empty `.ttl` files.
    - Excludes `.ttl` files exceeding a defined size threshold (default: 10 MB) to ensure compatibility with version
       control systems.
    - Skips files without corresponding SPARQL queries.

2. **Generates RDF/XML files**:
    - Converts `.ttl` files into `.rdf` (RDF/XML) format.
    - Integrates ontology information into the converted `.rdf` files.

3. **Validates and updates queries**:
    - Filters out queries that fail execution on the `.ttl` graphs.
    - Adapts SPARQL queries to match the new ontology structure.
    - Executes queries against the original `.ttl` and new `.rdf` files to ensure consistency of results.

### Failing queries

The reasons that cause queries to fail are syntax errors, which prevent the query from being interpreted or executed
 correctly due to invalid syntax. Sometimes, query errors occur due to a preprocessing failure (prefixes not being
 resolved properly) due to a graph mismatch (the query references entities from a different graph than the one
 declared). For example, here is a mismatched query from `raw/queries/dev/nl2sparql.json`, referencing airlines instead
 of cars (the actual KG content).
```JSON
{
    "kg_name": "car_1",
    "question": "What are the IDs and names of all countries that either have more than 3 car makers or produce Fiat models?",
    "query": "SELECT ?t1_country WHERE { ?t1 a :airlines . ?t1 :airlines\\#country ?t1_country . ?t1 :airlines\\#airline ?t1_airline . FILTER(?t1_airline = 'JetBlue Airways') . }"
}
```

When one of our adapted queries fails on the `.rdf` graph, we verify that the original query also fails on the original
 `.ttl` graph. This ensures that none of our adaptations introduced errors and that the failure originates from the
 source query.

## Execution

To start the preprocessing, ensure the `src` folder of this repository is in `PYTHONPATH`, that you downloaded and
 placed the original dataset in the `raw` folder, and then run `main.py` using the following command:
```bash
python main.py 2> output.log
```

## Output folder structure
After running the script, the processed files will be stored as:
```
processed/
├── graph/
│   ├── dev/
│   └── train/
├── queries/
│   ├── dev/
│   └── train/
```
