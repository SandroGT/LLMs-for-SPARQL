# Spider4SPARQL - Ontology Refinement

This README describes the ontology refinement step for the **Spider4SPARQL** dataset, which is designed for the
 **natural language to SPARQL translation task**. The ontology refinement step requires manual supervision of the RDF
graphs to improve the naming of classes and properties, ensuring clarity and consistency.

## Purpose

Some RDF graphs contain poorly named elements that may be misleading and confusing for both humans and automated
systems using these graphs. The main issues observed include:
- Many object properties follow a pattern like `ref-<relation>-id`, implying they reference an identifier rather than
-    another node. The `id` suffix is often misleading.
- Abbreviated or cryptic names that do not make the entity's semantics intuitive and explicit.

Some actions, such as removing unnecessary attributes introduced in preprocessing, are automated. However, updating
 names requires human supervision. The script iterates over all graphs and, for each one, prints the current ontology
to the command line, allowing the user to rename entities. It automatically updates SPARQL queries accordingly.

Updates are tracked with a logger and with `.git`.

## Execution

To start the refinement process, ensure the `src` folder of this repository is in `PYTHONPATH`, that you have executed
 the preprocessing script, and then run `main.py` using the following command:
```bash
python main.py 2> output.log
```

## Output folder structure

After running the script, the files from the preprocessing step will be overwritten. They can be found in:
```
processed/
├── graph/
│   ├── dev/
│   └── train/
├── queries/
│   ├── dev/
│   └── train/
```
