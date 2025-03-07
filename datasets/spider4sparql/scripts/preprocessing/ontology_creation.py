"""
Functions for populating Owlready2 ontologies from an RDFLib graph.

These functions create OWL entities, properties, and axioms to ensure the ontology accurately represents the structure
 and data of the RDF graph.
"""

import decimal
from pathlib import Path
import types

import owlready2 as owl
import rdflib as rdf

from uri_manipulation import get_stem_from_uri, label_from_stem, RdfOwlMap
from rdflib_processing import (RDF_NAMESPACE, RDF_TYPE, load_ttl_graph, get_rdf_classes, get_rdf_object_properties,
                               get_rdf_data_properties, get_rdf_individuals, get_rdf_range, get_rdf_domain)
from utils import get_python_name


def ttl_to_rdf(ttl_file: Path, store_path: Path = None) -> tuple[owl.Ontology, RdfOwlMap]:
    # Create Owlready2 ontology in dedicated world (each ontology will be isolated)
    base_iri = RDF_NAMESPACE
    world = owl.World()
    onto = world.get_ontology(base_iri)

    # Initializing map object to lately update queries (URI may be changed)
    map_obj = RdfOwlMap()

    # Load cleaned RDFlib graph
    rdflib_graph = load_ttl_graph(ttl_file)

    # Populate Owlready2 ontology with concepts from the RDFlib graph
    create_owl_classes(rdflib_graph, onto, map_obj)
    create_owl_object_properties(rdflib_graph, onto, map_obj)
    create_owl_data_properties(rdflib_graph, onto, map_obj)
    create_owl_individuals(rdflib_graph, onto, map_obj)
    create_owl_domain_range_axioms(rdflib_graph, onto, map_obj)
    create_triples(rdflib_graph, map_obj)  # Doesn't need any input ontology: it retrieves its entities from the map

    # Store the ontology and individuals in RDF/XML
    if store_path is not None:
        onto.save(file=str(store_path), format='rdfxml')

    return onto, map_obj


def create_owl_classes(graph: rdf.Graph, ontology: owl.Ontology, map_obj: RdfOwlMap):
    """Creates OWL classes from RDF graph data and adds them to the provided ontology.
    The mapping of the OWL class to its URI is also stored in the map."""
    with ontology:
        # Iterate over RDF classes extracted from the graph
        for c in get_rdf_classes(graph):
            # Split the URI of the class to extract its stem
            c_stem = get_stem_from_uri(c)
            # Ensure the class does not have a complex hierarchy (i.e., it has no type relations)
            assert not list(graph.triples((c, RDF_TYPE, None)))

            # Create a new OWL class
            onto_entity = types.new_class(c_stem, (owl.Thing,))
            onto_entity.label = label_from_stem(c_stem, capitalize=True)

            # Add the created class to the map for later use
            map_obj.add_mapping(onto_entity, c)


def create_owl_object_properties(graph: rdf.Graph, ontology: owl.Ontology, map_obj: RdfOwlMap):
    """Creates OWL object properties from RDF graph data and adds them to the provided ontology.
    The mapping of the OWL object property to its URI is also stored in the map."""
    with ontology:
        # Get the RDFlib object properties
        rdf_object_properties = get_rdf_object_properties(graph)

        # Get the stem to use for each object property
        property_stem_map = generate_refined_stem_map(graph, rdf_object_properties)

        # Iterate over RDF object properties extracted from the graph
        for p in get_rdf_object_properties(graph):
            # Get the stem to use for the current property
            p_stem = property_stem_map[p]

            # Create a new OWL object property (or retrieve the one with the same stem, if already existing)
            onto_entity = types.new_class(p_stem, (owl.ObjectProperty,))
            onto_entity.python_name = get_python_name(onto_entity.python_name)
            onto_entity.label = label_from_stem(p_stem, capitalize=False)

            # Add the created object property to the map for later use
            map_obj.add_mapping(onto_entity, p)


def create_owl_data_properties(graph: rdf.Graph, ontology: owl.Ontology, map_obj: RdfOwlMap):
    """Creates OWL data properties from RDF graph data and adds them to the provided ontology.
    The mapping of the OWL data property to its URI is also stored in the map."""
    with ontology:
        # Get the RDFlib data properties
        rdf_data_properties = get_rdf_data_properties(graph)

        # Get the stem to use for each data property
        property_stem_map = generate_refined_stem_map(graph, rdf_data_properties)

        # Iterate over RDF data properties extracted from the graph
        for a in rdf_data_properties:
            # Get the stem to use for the current property
            a_stem = property_stem_map[a]

            # Create a new OWL data property (or retrieve the one with the same stem, if already existing)
            onto_entity = types.new_class(a_stem, (owl.DataProperty,))
            onto_entity.python_name = get_python_name(onto_entity.python_name)
            onto_entity.label = label_from_stem(a_stem, capitalize=False)

            # Add the created data property to the map for later use
            map_obj.add_mapping(onto_entity, a)


def generate_refined_stem_map(graph: rdf.Graph, properties: set[rdf.URIRef]) -> dict[rdf.URIRef, str]:
    """Generates a mapping of RDF properties to their refined stems, ensuring that only properties with the same base
    stem and same range are grouped under a shared stem."""
    # Map each property to its range (the type of value it links to)
    property_range_map = {
        prop: str(sorted([
            str(t) for t in get_rdf_range(graph, prop)
        ]))
        for prop in properties
    }

    # Group properties by their "base" stem (extracted from the URI)
    base_stem_groups = {}
    for prop in properties:
        base_stem = get_stem_from_uri(prop, separators='/#')  # Extract base stem
        if base_stem in base_stem_groups:
            base_stem_groups[base_stem].add(prop)
        else:
            base_stem_groups[base_stem] = {prop}

    # Adjust stems when multiple properties share the same base stem but have different ranges
    refined_stem_map = {}
    for base_stem, prop_set in base_stem_groups.items():
        property_ranges = [property_range_map[prop] for prop in prop_set]

        if len(set(property_ranges)) == 1:
            # All properties in this group have the same range, keep the base stem
            for prop in prop_set:
                assert prop not in refined_stem_map
                refined_stem_map[prop] = base_stem
        else:
            # Different ranges require a more specific stem
            for prop in prop_set:
                refined_stem = get_stem_from_uri(prop, separators='/').replace('#', '-')  # Extract a more detailed stem
                assert prop not in refined_stem_map
                refined_stem_map[prop] = refined_stem

    return refined_stem_map


def create_single_owl_individual(
        rdf_individual: rdf.URIRef | rdf.BNode, graph: rdf.Graph, ontology: owl.Ontology
) -> owl.Thing:
    """Creates a single OWL individual from an RDF individual."""
    # Determine the types of the individual
    types_set = sorted({o for s, p, o in graph.triples((rdf_individual, RDF_TYPE, None))})
    assert types_set
    # Get all ontology classes
    owl_types = [get_entity_by_iri(type_uri, ontology) for type_uri in types_set]
    # Get the relative URI stem for the individual
    if isinstance(rdf_individual, rdf.URIRef):
        individual_uri_stem = get_stem_from_uri(rdf_individual)
        class_uri_part = '-'.join([t.name for t in owl_types])
        relative_iri = f'{class_uri_part}/{individual_uri_stem}'
    else:
        assert isinstance(rdf_individual, rdf.BNode)
        relative_iri = None
    with ontology:
        owl_class = owl_types[0]
        onto_entity = owl_class(relative_iri)
        for owl_class in owl_types[1:]:
            onto_entity.is_a.append(owl_class)
    # Return the created individual
    return onto_entity


def create_owl_individuals(graph: rdf.Graph, ontology: owl.Ontology, map_obj: RdfOwlMap):
    """Creates OWL individuals from RDF graph data and adds them to the provided ontology.
    The mapping of the OWL individual to its URI is stored in the map."""
    # Extract named and anonymous individuals from the RDF graph
    named_individuals, anonymous_individuals = get_rdf_individuals(graph)

    # Process named individuals
    for i in named_individuals | anonymous_individuals:
        # Create the OWL individual
        onto_entity = create_single_owl_individual(i, graph, ontology)
        # Add the created individual to the map for later use
        map_obj.add_mapping(onto_entity, i)


def create_owl_domain_range_axioms(graph: rdf.Graph, ontology: owl.Ontology, map_obj: RdfOwlMap):
    """Creates domain and range axioms for object and data properties in the ontology.
    It processes the object and data properties in the ontology, retrieves the corresponding RDF domain and range
    information from the RDF graph, and constructs the appropriate OWL domain and range axioms using the mapping
    between RDF and OWL entities."""
    with ontology:
        # Process each object property in the ontology
        for p in ontology.object_properties():
            # Object property domain: get the RDF domain types for the property
            p_rdf_domain = {
                type_uri
                for rdf_uri in map_obj.to_rdf(p)
                for type_uri in get_rdf_domain(graph, rdf_uri)
            }
            assert all({isinstance(type_uri, rdf.URIRef) for type_uri in p_rdf_domain})  # Ensure domain types are URIs
            p_owl_domain = [
                map_obj.to_owl(type_uri)
                for type_uri in p_rdf_domain
            ]  # Convert RDF types to OWL classes
            p.domain = [owl.class_construct.Or(p_owl_domain)]  # Set the domain as an 'Or' construct for the property

            # Object property range: get the RDF range types for the property
            p_rdf_range = {
                type_uri
                for rdf_uri in map_obj.to_rdf(p)
                for type_uri in get_rdf_range(graph, rdf_uri)
            }
            assert all({isinstance(type_uri, rdf.URIRef) for type_uri in p_rdf_range})  # Ensure range types are URIs
            p_owl_range = [
                map_obj.to_owl(type_uri)
                for type_uri in p_rdf_range
            ]  # Convert RDF types to OWL classes
            p.range = [owl.class_construct.Or(p_owl_range)]  # Set the range as an 'Or' construct for the property

        # Process each data property in the ontology
        for a in ontology.data_properties():
            # Data property domain: get the RDF domain types for the property
            a_rdf_domain = {
                type_uri
                for rdf_uri in map_obj.to_rdf(a)
                for type_uri in get_rdf_domain(graph, rdf_uri)
            }
            assert all({isinstance(type_uri, rdf.URIRef) for type_uri in a_rdf_domain})  # Ensure domain types are URIs
            a_owl_domain = [
                map_obj.to_owl(type_uri)
                for type_uri in a_rdf_domain
            ]  # Convert RDF types to OWL classes
            a.domain = [owl.class_construct.Or(a_owl_domain)]  # Set the domain as an 'Or' construct for the property

            # Data property range: get the RDF range for the property (could be raw data types)
            a_raw_range = {
                normalize_raw_type(raw_type)
                for rdf_uri in map_obj.to_rdf(a)
                for raw_type in get_rdf_range(graph, rdf_uri)
            }
            assert all(
                {isinstance(raw_type, type) for raw_type in a_raw_range})  # Ensure raw types are valid Python types
            a.range = [owl.class_construct.Or(a_raw_range)]  # Set the range as an 'Or' construct for the property


def create_triples(graph: rdf.Graph, map_obj: RdfOwlMap):
    """Populates an ontology with data from an RDF graph by iterating over triples and setting the values on the
    corresponding OWL entities."""
    for s, p, o in graph:
        # Retrieve the OWL entities for the subject and predicate from the map
        s_entity = map_obj.to_owl(s)
        p_entity = map_obj.to_owl(p)

        # If either subject or predicate are not in the mapping, skip this triple
        if not (s_entity and p_entity):
            continue

        # Handle object based on its type: URI, blank node, or literal
        if isinstance(o, rdf.URIRef) or isinstance(o, rdf.BNode):
            o_to_store = map_obj.to_owl(o)  # Resolve to OWL entity
        else:
            assert isinstance(o, rdf.Literal)  # Ensure object is a literal
            o_to_store = normalize_raw_value(o.value)  # Extract value from the literal

        # Access the property in the subject entity and set its value
        p_value = getattr(s_entity, p_entity.python_name)

        # If the property value is a list, append the object to the list; otherwise, set the property value directly
        if isinstance(p_value, list):
            if o_to_store not in p_value:
                p_value.append(o_to_store)
        else:
            if not hasattr(s_entity, p_entity.python_name):
                setattr(s_entity, p_entity.python_name, o_to_store)
            else:
                stored_o = getattr(s_entity, p_entity.python_name)
                assert stored_o and stored_o == o_to_store, (stored_o, o_to_store,)


def get_entity_by_iri(iri: str | rdf.URIRef, onto: owl.Ontology) -> owl.ThingClass:
    """Retrieve a single entity from the Owlready2 ontology based on its IRI."""
    if isinstance(iri, rdf.URIRef):
        iri = str(iri)

    entity_list = onto.search(iri=iri)
    if not entity_list:
        raise RuntimeError(f"No entities found for IRI: {iri}")
    if len(entity_list) > 1:
        raise RuntimeError(f"IRI '{iri}' is ambiguous; multiple entities match the IRI.")

    return entity_list[0]


def normalize_raw_type(t: type):
    """Normalizes raw literal types to ensure compatibility with Python types and Owlready2 storage."""
    assert isinstance(t, type)

    if t is decimal.Decimal:
        return float
    else:
        return t


def normalize_raw_value(v):
    """Normalizes raw literal values to ensure quality and compatibility with Python types and Owlready2 storage."""
    if isinstance(v, decimal.Decimal):
        return float(v)
    elif isinstance(v, str):
        # Some string values inexplicably contain many trailing spaces, resulting in excessive '%20' values in their
        #  URIs. These trailing spaces are absent in queries, causing discrepancies and issues.
        return v.strip()
    else:
        return v
