from pathlib import Path
import shutil

import pandas as pd
import rdflib as rdf
from tqdm import tqdm

from logger import LOGGER
from uri_manipulation import RdfOwlMap
from ontology_creation import ttl_to_rdf
from query_processing import check_mentions, preprocess_ttl_query, convert_to_rdf_query, safe_query
from utils import convert_bytes

SCRIPT_PATH = Path(__file__).parent.resolve()
RAW_PATH = SCRIPT_PATH.joinpath('..', '..', 'raw').resolve()
PROCESSED_PATH = SCRIPT_PATH.joinpath('..', '..', 'processed').resolve()

QUERY_FOLDER = 'queries'
QUERY_FILE = 'nl2sparql'
OWL_FOLDER = 'owl'
TTL_FOLDER = 'ttl'
GRAPH_FOLDER = 'graph'

FOLDS = ['dev', 'train']

GRAPH_SIZE_THRESHOLD = 10 * 1024 ** 2  # Threshold to 10MB (keeps git working well)


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "The raw data folder does not exist. Please ensure the dataset is in the correct location."
        )

    # Create the output folders
    LOGGER.info(f"Creating output folders in '{PROCESSED_PATH}'.")
    if PROCESSED_PATH.exists():
        shutil.rmtree(PROCESSED_PATH)
    for fold in FOLDS:
        query_folder = PROCESSED_PATH.joinpath(QUERY_FOLDER, fold)
        query_folder.mkdir(parents=True)
        graph_folder = PROCESSED_PATH.joinpath(GRAPH_FOLDER, fold)
        graph_folder.mkdir(parents=True)

    # Select raw graphs to process
    LOGGER.info(f"Selecting raw graphs from '{RAW_PATH}' for processing.")
    ttl_paths_info = filter_graphs()

    # Create ontologies
    LOGGER.info(f"Converting selected raw graphs to RDF/XML.")
    ttl_rdf_data_map = convert_graphs(ttl_paths_info)

    # Check ontologies and update queries
    LOGGER.info(f"Updating and checking queries.")
    update_queries(ttl_paths_info, ttl_rdf_data_map)


def filter_graphs() -> list[tuple[str, Path]]:
    """Filters RDF graph files based on size, queries, and validity.

    Processes TTL graph files located in specified dataset directories and applies the following filters:
    - Excludes files that are empty.
    - Excludes files exceeding a specified size threshold.
    - Excludes files without associated queries.
    - Excludes files with syntax errors during parsing.
    """
    LOGGER.info(f"Graph size threshold set to {convert_bytes(GRAPH_SIZE_THRESHOLD)}.")

    # Select the TTL files to keep
    ignored = 0
    kept = 0
    ttl_paths = list()

    # Iterate through each fold (subdirectory in the dataset)
    for fold in FOLDS:
        query_path = RAW_PATH.joinpath(QUERY_FOLDER, fold, f'{QUERY_FILE}.csv')
        assert query_path.exists()
        query_df = pd.read_csv(query_path)

        # Iterate through each TTL file in the current fold's directory
        for raw_ttl_path in sorted(RAW_PATH.joinpath(TTL_FOLDER, fold).iterdir()):
            # Get the file size
            file_size = raw_ttl_path.stat().st_size

            # Get the graph queries
            g_name = raw_ttl_path.stem
            g_query_df = query_df[query_df['kg_name'] == g_name]

            # Flag to determine if the current file should be skipped
            skip = False

            # Skip empty files
            if file_size == 0:
                LOGGER.debug(f"[{fold}] Ignored KG '{g_name}': empty graph.")
                skip = True
            # Skip files that exceed the size threshold
            elif file_size > GRAPH_SIZE_THRESHOLD:
                LOGGER.debug(f"[{fold}] Ignored KG '{g_name}': too large ({convert_bytes(file_size)}).")
                skip = True
            # Skip graph for which there are no queries
            elif len(g_query_df) == 0:
                LOGGER.debug(f"[{fold}] Ignored KG '{g_name}': no queries available.")
                skip = True
            # Try parsing the file, skip if there's a syntax error
            else:
                try:
                    g = rdf.Graph()
                    g.parse(str(raw_ttl_path))
                except SyntaxError:
                    LOGGER.debug(f"[{fold}] Ignored KG '{g_name}': invalid syntax.")
                    skip = True

            # Update counters based on whether the file is skipped or not
            if skip:
                ignored += 1
            else:
                ttl_paths.append((fold, raw_ttl_path,))
                kept += 1

    LOGGER.info(f"Ignored {ignored} KGs, kept {kept} KGs.")
    return ttl_paths


def convert_graphs(ttl_paths_info: list[tuple[str, Path]]) -> dict[Path, tuple[Path, RdfOwlMap]]:
    """Converts TTL files to RDF/XML format and maps RDFlib entities to Owlready2 entities.

    Processes a list of TTL file paths, converts each file to RDF/XML format, and generates a mapping of RDFlib
    entities to Owlready2 entities. The converted RDF files are stored in a specified directory structure.

    Returns a dictionary that allows retrieval of the new RDF/XML path and the entities map from the TTL path.
    """
    # Prepare output dictionary
    output_map: dict[Path, tuple[Path, RdfOwlMap]] = dict()

    # Iterate through each TTL file
    for i, (fold, ttl_file) in enumerate(ttl_paths_info):
        g_name = ttl_file.stem
        LOGGER.debug(f"{i} - [{fold}] Processing graph '{g_name}'.")
        rdf_store_path = PROCESSED_PATH.joinpath(GRAPH_FOLDER, fold, f'{g_name}.rdf')
        # Convert TTL to RDF/XML and get the entities map (RDFlib URIs to Owlready2 and vice versa)
        _, map_obj = ttl_to_rdf(ttl_file, rdf_store_path)
        # Update the output
        output_map[ttl_file] = (rdf_store_path, map_obj)

    return output_map


def update_queries(
        ttl_paths_info: list[tuple[str, Path]], ttl_rdf_data_map: dict[Path, tuple[Path, RdfOwlMap]]
) -> None:
    """Processes and validates SPARQL queries for different knowledge graphs (KGs), comparing query runs from
    original TTL and adapted RDF/XML graphs.

    1. Loads SPARQL queries from CSV files corresponding to each knowledge graph.
    2. For each TTL and RDF/XML graph pair, adapts the SPARQL queries by converting TTL URIs to Owlready2 entity URIs.
    3. Executes the queries against both the old TTL graph and the new RDF/XML graph.
    4. Compares runs from both graphs to ensure they are consistent.
    5. If any query fails in the RDF graph, it is marked as a failed query and discarded.
    6. Stores successful queries, in their partial and complete URIs formats, into a CSV file for each knowledge graph.
    """
    # Load queries
    query_paths = {fold: RAW_PATH.joinpath(QUERY_FOLDER, fold, f'{QUERY_FILE}.csv') for fold in FOLDS}
    query_dfs = {fold: pd.read_csv(p) for fold, p in query_paths.items()}

    # Iterate through each TTL and RDF/XML file
    for i, (fold, ttl_file) in enumerate(ttl_paths_info):
        rdf_file, map_obj = ttl_rdf_data_map[ttl_file]
        all_ttl_uris = [rdf_uri for rdf_uri in map_obj.rdf2owl.keys() if rdf_uri.startswith('http')]
        g_name = ttl_file.stem

        # Select graph queries
        df_fold = query_dfs[fold]
        df_graph = df_fold[df_fold['kg_name'] == g_name]
        queries = df_graph['query'].to_list()

        LOGGER.info(f"{i} - [{fold}] Checking '{g_name}' KG: {len(queries)} queries found.")

        # Validate query assumptions
        check_mentions(ttl_file, queries)

        # Test queries
        failed_queries = 0
        new_df_queries = pd.DataFrame(columns=['nl_question', 'sparql_partial_uri', 'sparql_complete_uri'])

        with tqdm(df_graph.iterrows(), total=len(queries), unit='query', postfix={'Failed': failed_queries}) as pbar:
            for row_idx, (_, nl_question, ttl_query) in pbar:
                # Execute original query in old TTL graph
                q_ttl = preprocess_ttl_query(ttl_query, all_ttl_uris)
                r_ttl, n_ttl, f_ttl, e_ttl = safe_query(ttl_file, q_ttl)

                # Execute adapted query in new RDF/XML graph
                q_rdf_complete, q_rdf_partial = convert_to_rdf_query(q_ttl, map_obj)
                r_rdf, n_rdf, f_rdf, e_rdf = safe_query(rdf_file, q_rdf_partial)

                # If RDF did not fail
                if not f_rdf:
                    # If TTL also did not fail and 'filter' is not in the query ...
                    if not f_ttl and 'filter' not in q_ttl.lower():
                        # ... then we expect the same number of runs
                        assert n_rdf == n_ttl
                    # Track the query
                    new_df_queries.loc[len(new_df_queries)] = (nl_question, q_rdf_partial, q_rdf_complete,)

                # If RDF failed
                elif f_rdf:
                    assert f_ttl  # TTL should have failed too
                    assert type(e_ttl) is type(e_rdf)  # Ensure the exceptions are of the same type
                    failed_queries += 1  # Count as a non-runnable query

                # Update failed queries count in progress bar
                pbar.set_postfix({'Failed': failed_queries})

        # Store each KG queries in a separate file (not of a unique file for the entire fold as in the original dataset)
        output_df_path = PROCESSED_PATH.joinpath(QUERY_FOLDER, fold, f'{g_name}.csv')
        new_df_queries.to_csv(output_df_path, index=False)


if __name__ == '__main__':
    main()
