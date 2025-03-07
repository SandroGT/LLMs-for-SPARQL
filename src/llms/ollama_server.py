import os

from ollama import Client, Options

from llms.abstract import InstructedLLM


class OllamaServerLLM(InstructedLLM):
    _client: Client

    def __init__(
            self,
            model_id: str,
    ):
        if 'OLLAMA_HOST' not in os.environ:
            raise ValueError('OLLAMA_HOST environmental variable must be set')

        self.model_id = model_id
        self._client = Client(host=os.environ['OLLAMA_HOST'])

    def chat(
            self,
            messages: list[dict],
            max_new_tokens: int = 256,
            temperature: float = 0.6,
            top_p: float = 0.9,
            terminators: list[str] = None,
            **_,
    ) -> str:
        options = Options(temperature=temperature, num_predict=max_new_tokens, top_p=top_p, stop=terminators)
        response = self._client.chat(model=self.model_id, messages=messages, options=options, keep_alive=-1)
        return response['message']['content']
