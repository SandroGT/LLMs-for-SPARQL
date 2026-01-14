from __future__ import annotations
from typing import Iterable, Generator
from evaluation.query_types import QueryCategory


class RunResults:
    """Represents the results of a run, which includes query results over multiple datasets, graphs, prompt types,
     and iterations."""
    datasets_data: dict[str, DatasetData]

    def __init__(self, run_dict: dict):
        """Initializes the RunResults class with data from a run."""
        # Organize data by dataset name
        self.datasets_data = {
            dataset_name: DatasetData(dataset_dict)
            for dataset_name, dataset_dict in run_dict.items()
        }

    def __iter__(self) -> Iterable[tuple[str, DatasetData]]:
        """Iterator over the dataset data."""
        return iter(self.datasets_data.items())

    def retrieve(
            self,
            datasets: str | set[str] = None,
            graphs: str | set[str] = None,
            prompts: str | set[str] = None,
            query_categories: QueryCategory | set[QueryCategory] = None,
            return_iterations: bool = False,
            most_voted: bool = True,
            default_iteration: int = None
    ) -> list[QueryExecutionData | IterationData | None]:
        """Retrieves query data based on given filters (e.g., datasets, graphs, prompts, etc.).

        Args:
            datasets (str | set[str], optional): Dataset names to filter.
            graphs (str | set[str], optional): Graph names to filter.
            prompts (str | set[str], optional): Prompt names to filter.
            query_categories (QueryCategory | set[QueryCategory], optional): Query categories to filter.
            return_iterations (bool, optional): Whether to yield IterationData or QueryExecutionData.
            most_voted (bool, optional): Whether to yield the most voted QueryExecutionData from each iteration, or all
             the QueryExecutionData in them.
            default_iteration (bool, int): When there is not a most voted, the iteration to return.

        Yields:
            QueryExecutionData | IterationData: filtered data based on the provided arguments.
        """
        # Convert single string inputs into sets for uniform handling
        if isinstance(datasets, str):
            datasets = {datasets}

        # Iterate through datasets and yield relevant data
        return [
            result
            for dataset_name, dataset_data in self
            for result in dataset_data.retrieve(
                graphs, prompts, query_categories, return_iterations, most_voted, default_iteration
            )
            if datasets is None or dataset_name in datasets
        ]


class DatasetData:
    """Represents the data for a specific dataset, including graphs and queries."""
    graphs_data: dict[str, GraphData]

    def __init__(self, dataset_dict: dict):
        """Initializes the DatasetData class with data for multiple graphs."""
        # Organize data by graph name
        self.graphs_data = {
            graph_name: GraphData(query_list)
            for graph_name, query_list in dataset_dict.items()
        }

    def __iter__(self) -> Iterable[tuple[str, GraphData]]:
        """Iterator over the graph data."""
        return iter(self.graphs_data.items())

    def retrieve(
            self,
            graphs: str | set[str],
            prompts: str | set[str],
            query_categories: QueryCategory | set[QueryCategory],
            return_iterations: bool,  # if true, returns Generator[IterationData]
            most_voted: bool,
            default_iteration: int = None
    ) -> Generator[QueryExecutionData | IterationData | None]:
        """Retrieves query data for specific graphs based on the given filters."""
        # Convert single string inputs into sets for uniform handling
        if isinstance(graphs, str):
            graphs = {graphs}

        # Iterate through graphs and yield relevant data
        for graph_name, graph_data in self:
            if graphs is None or graph_name in graphs:
                yield from graph_data.retrieve(
                    prompts, query_categories, return_iterations, most_voted, default_iteration
                )


class GraphData:
    """Represents the data for a specific graph, including associated query data."""
    queries_data: list[QueryData]

    def __init__(self, query_list: list):
        """Initializes the GraphData class with query data."""
        # Organize query data into QueryData objects
        self.queries_data = [QueryData(query_dict) for query_dict in query_list]

    def __iter__(self) -> Iterable[QueryData]:
        """Iterator over the query data."""
        return iter(self.queries_data)

    def retrieve(
            self,
            prompts: str | set[str],
            query_categories: QueryCategory | set[QueryCategory],
            return_iterations: bool,  # if true, returns Generator[IterationData]
            most_voted: bool,
            default_iteration: int = None
    ) -> Generator[QueryExecutionData | IterationData | None]:
        """Retrieves query execution or iteration data based on filters for prompts and query categories."""
        for query_data in self:
            yield from query_data.retrieve(
                prompts, query_categories, return_iterations, most_voted, default_iteration
            )


class QueryData:
    """Represents the data for a specific query, including its iterations and categories."""
    query_id: int
    categories: set[QueryCategory]
    iteration_data: dict[str, IterationData]

    def __init__(self, query_dict: dict):
        """Initializes the QueryData class with data for a specific query."""
        self.query_id = query_dict['id']
        self.categories = set(query_dict['categories'])
        # Organize iteration data by prompt name
        self.iteration_data = {
            prompt_name: IterationData(prompt_dict)
            for prompt_name, prompt_dict in query_dict['evaluation'].items()
        }

    def __iter__(self) -> Iterable[tuple[str, IterationData]]:
        """Iterator over the iteration data."""
        return iter(self.iteration_data.items())

    def retrieve(
            self,
            prompts: str | set[str],
            query_categories: QueryCategory | set[QueryCategory],
            return_iterations: bool,  # if true, returns Generator[IterationData]
            most_voted: bool,
            default_iteration: int = None
    ) -> Generator[QueryExecutionData | IterationData | None]:
        """Retrieves iteration data or query execution data based on filters for prompts and categories."""
        # Ensure single values are treated as sets
        if isinstance(prompts, str):
            prompts = {prompts}
        if isinstance(query_categories, QueryCategory):
            query_categories = {query_categories}

        # Filter by prompts and categories, then yield data
        for prompt_name, iteration_data in self:
            prompt_match = (prompts is None or prompt_name in prompts)
            category_match = (query_categories is None or query_categories & self.categories)
            if prompt_match and category_match:
                if return_iterations:
                    yield iteration_data
                else:
                    yield from iteration_data.retrieve(most_voted, default_iteration)


class IterationData:
    """Represents data for a specific iteration of a query execution."""
    groups: list[list[int]]
    most_voted_id: int
    repetitions_data: list[QueryExecutionData]

    def __init__(self, query_dict: dict):
        """Initializes the IterationData class with data for a specific iteration."""
        self.groups = query_dict['groups']
        self.most_voted_id = query_dict['most_voted_id']
        # Organize repetition data into QueryExecutionData objects
        self.repetitions_data = [QueryExecutionData(repetition_dict) for repetition_dict in query_dict['repetitions']]

    def __iter__(self) -> Iterable[QueryExecutionData]:
        """Iterator over the repetition data."""
        return iter(self.repetitions_data)

    def retrieve(
            self, most_voted: bool, default_iteration: int = None
    ) -> Generator[QueryExecutionData | IterationData | None]:
        """Retrieves either the most voted iteration or all repetitions."""
        if most_voted:
            if self.most_voted_id is not None:
                yield self.repetitions_data[self.most_voted_id]
            elif default_iteration is not None:
                yield self.repetitions_data[default_iteration]
            else:
                yield None
        else:
            for query_execution_data in self.repetitions_data:
                yield query_execution_data


class QueryExecutionData:
    """Represents data for a single query execution, including timing and correctness."""
    time: float
    generation_error: bool
    execution_error: bool
    is_correct: bool

    def __init__(self, repetition_dict: dict):
        """Initializes the QueryExecutionData class with repetition-specific data."""
        self.time = repetition_dict.get('generation_time')
        self.generation_error = repetition_dict.get('parsing_error', False)
        self.execution_error = repetition_dict.get('execution_error')
        self.is_correct = repetition_dict.get('is_correct')
