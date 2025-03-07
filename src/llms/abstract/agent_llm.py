from abc import ABC, abstractmethod
from typing import Any

from llms.abstract import InstructedLLM


class AgentLLM(ABC):
    _SYSTEM_TEMPLATE: str
    _USER_TEMPLATE: str
    _EXAMPLES_DATA: list[tuple[str, str]]

    _llm: InstructedLLM
    _llm_settings: dict

    def __init__(self, llm: InstructedLLM, llm_settings: dict = None):
        if not hasattr(self, '_SYSTEM_TEMPLATE'):
            raise NotImplementedError('Attribute `SYSTEM_TEMPLATE` has not been set.')
        if not hasattr(self, '_USER_TEMPLATE'):
            raise NotImplementedError('Attribute `USER_TEMPLATE` has not been set.')
        if not hasattr(self, '_EXAMPLES_DATA'):
            self._EXAMPLES_DATA = list()

        self._llm = llm
        self._llm_settings = llm_settings if llm_settings is not None else dict()

    @abstractmethod
    def _format_system(self, **system_data) -> str:
        pass

    @abstractmethod
    def _format_user(self, **user_data) -> str:
        pass

    @abstractmethod
    def _parse_answer(self, **parsing_data) -> Any:
        pass

    def run(
            self,
            user_data: dict,
            system_data: dict | None = None,
            parsing_data: dict | None = None,
    ) -> tuple[str, Any]:
        if system_data is None:
            system_data = dict()
        if parsing_data is None:
            parsing_data = dict()
        system = self._format_system(**system_data)
        user = self._format_user(**user_data)
        raw_answer = self._llm.task(system, user, self._EXAMPLES_DATA, **self._llm_settings)
        parsing_data = {'answer': raw_answer, **parsing_data}
        parsed_answer = self._parse_answer(**parsing_data)
        return raw_answer, parsed_answer
