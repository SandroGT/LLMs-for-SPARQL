import json
from pathlib import Path
import re

import owlready2 as owl
import pandas as pd

from logger import LOGGER
from refinement import remove_id_suffixes, interactive_ontology_supervision, update_query, get_onto_entities


SCRIPT_PATH = Path(__file__).parent.resolve()
DATA_PATH = SCRIPT_PATH.joinpath('..', '..', 'processed').resolve()
TRACKING_FILE = SCRIPT_PATH.joinpath('track.json')

QUERY_FOLDER = 'queries'
GRAPH_FOLDER = 'graph'

FOLDS = ['dev', 'train']


def main():
    # Keep track of graphs that have already been supervised
    supervised_kg_names: list[str] = list()

    # Load previously supervised graphs if a tracking file exists
    if TRACKING_FILE.exists():
        with TRACKING_FILE.open('r', encoding='utf8') as f:
            supervised_kg_names = json.load(f)

    # Iterate over all graphs
    for fold in FOLDS:
        for graph_path in sorted(DATA_PATH.joinpath(GRAPH_FOLDER, fold).iterdir()):
            g_name = graph_path.stem

            # Skip graphs that have already been supervised
            if f'{fold}/{g_name}' in supervised_kg_names:
                continue

            # Path to the associated SPARQL queries CSV file
            query_path = DATA_PATH.joinpath(QUERY_FOLDER, fold, f'{g_name}.csv')

            LOGGER.info(f"[{fold}] Supervising graph '{g_name}'.")

            # Load the RDF graph and the corresponding SPARQL query dataframe
            world = owl.World()
            graph = world.get_ontology(str(graph_path)).load()
            query_df = pd.read_csv(query_path)
            onto_entities = get_onto_entities(graph)

            # Create a mapping from current URIs to updated URIs (starts as an identity)
            uri_map = {e.iri: e.iri for e in onto_entities}

            # Update object property names by removing redundant "id" suffixes
            remove_id_suffixes(graph, uri_map)

            # Manually supervise and update the entity names
            interactive_ontology_supervision(graph, uri_map)

            # Update SPARQL queries to reflect the new URIs
            query_df['sparql_partial_uri'] = query_df['sparql_partial_uri'].apply(
                lambda x: update_query(x, uri_map, partial_uri=True)
            )
            query_df['sparql_complete_uri'] = query_df['sparql_complete_uri'].apply(
                lambda x: update_query(x, uri_map, partial_uri=False)
            )

            # Save the updated RDF graph and query CSV
            query_df.to_csv(query_path, index=False)
            save_cleaned_rdf(graph, graph_path)

            # Record the supervised graph to avoid reprocessing it next time
            supervised_kg_names.append(f'{fold}/{g_name}')
            with TRACKING_FILE.open('w', encoding='utf8') as f:
                json.dump(supervised_kg_names, f)

            LOGGER.info(f"[{fold}] Graph '{g_name}' supervised: updated .rdf graph and .csv queries.")

def save_cleaned_rdf(graph: owl.Ontology, graph_path: Path):
    """Saves an RDF/XML ontology file removing any owlready2 `<owlr:python_name>` attributes from it."""
    # Normally save the ontology to the file
    graph.save(file=str(graph_path), format='rdfxml')

    # Read the saved RDF/XML file
    with graph_path.open('r', encoding='utf8') as f:
        lines = f.readlines()

    # Remove lines containing `<owlr:python_name>` attributes
    cleaned_lines = [l for l in lines if not re.match(r'\s*<owlr:python_name .*', l)]

    # Overwrite the file again with the cleaned content
    with graph_path.open('w', encoding='utf8') as f:
        f.writelines(cleaned_lines)


if __name__ == '__main__':
    main()
