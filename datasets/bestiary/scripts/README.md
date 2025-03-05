# Beastiary - Ontology enhancement

This README describes the **ontology enhancement** step for the **Beastiary** dataset. The enhancement step focuses on
 improving the classification and labeling of beasts (entities) in the ontology, enriching it with meaningful categories
 and human-readable names for better clarity and consistency.

## Purpose

The **Beastiary** dataset consists of an ontology that categorizes different mythical and fantastical creatures.
 However, the ontology may suffer from ambiguous or inconsistent naming of classes, properties, and individual entities.
 The enhancement step addresses the following tasks:
- **Beast type classification**: adds relevant categories (e.g., "dragons", "robots") to individual entities based on
   their URIs using the `beast_category` data property.
- **Name addition**: adds names to individual entities based on their URIs using the `has_name` data property.
- **Human-readable labels**: assigns proper labels to ontology entities.
- **Query testing**: filter the queries, keeping only those that successfully run on the respective graph.

These additions are primarily made to avoid operations on the URI at query and generation time (e.g., filtering using
 regex on URIs), which are not ideal for efficient querying. The script ensures that the relevant information is
 pre-processed and stored as properties, improving the ontology's usability and query performance.

## Execution

To start the enhancement process, ensure that you have the dataset and necessary directories set up, and then execute
 the following command to run the main process:

```bash
python enchancement.py
```

# Output folder structure

After running the script, the enhanced files will be saved in the following directory structure:
```
processed/
├── graph/
│   └── dev/beastiary.rdf
├── queries/
│   └── dev/beastiary.csv
```
