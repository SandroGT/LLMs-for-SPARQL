import os

from openai import OpenAI

from llms.abstract import InstructedLLM


class GPTFamilyLLM(InstructedLLM):
    _client: OpenAI

    def __init__(
            self,
            model_id: str
    ):
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError('OPENAI_API_KEY environmental variable must be set')
        self.model_id = model_id
        self._client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    def chat(
            self,
            messages: list[dict],
            max_new_tokens: int = 256,
            temperature: float = 0.6,
            top_p: float = 0.9,
            terminators: list[str] = None,
            **_,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=terminators,
        )
        assert len(response.choices) == 1
        response = response.choices[0].message.content
        return response
