from abc import ABC, abstractmethod


class InstructedLLM(ABC):
    """A causal LLM fine-tuned to follow instructions, designed to execute tasks through prompting."""
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def chat(
            self,
            messages: list[dict],  # TODO add support for list[list[dict]] (so, support for batches)
            max_new_tokens: int = 256,
            temperature: float = 0.6,
            top_p: float = 0.9,
            terminators: list[str] = None
    ) -> str:
        pass

    def task(
            self,
            system: str,
            user: str,  # TODO add support for list[str] (so, support for batches)
            examples: list[tuple[str, str]] = None,
            max_new_tokens: int = 256,
            temperature: float = 0.01,
            top_p: float = 0.1,
            terminators: list[str] = None,
    ) -> str:
        instruct_messages = [
            {'role': 'system', 'content': system},
            *[{'role': r, 'content': m} for e in examples for r, m in zip(['user', 'assistant'], e)],
            {'role': 'user', 'content': user},
        ]
        return self.chat(instruct_messages, max_new_tokens, temperature, top_p, terminators=terminators)
