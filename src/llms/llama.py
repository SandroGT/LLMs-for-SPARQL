from transformers import BitsAndBytesConfig

from llms.transformers import TransformersLLM


class LlamaFamilyLLM(TransformersLLM):

    def __init__(self, model_id: str, quantization_config: BitsAndBytesConfig = None):
        super().__init__(model_id, quantization_config)
        self._tokenizer.mask_token = '<mask>'
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        # SEE https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/discussions/2
        self._base_terminators.append(self._tokenizer.convert_tokens_to_ids('<|eot_id|>'))


class Llama2(LlamaFamilyLLM):
    _AVAILABLE_VERSIONS: list = [7, 13, 70]

    def __init__(
            self,
            model_version: int = 7,
            quantization_config: BitsAndBytesConfig = None,
    ):
        if model_version not in self._AVAILABLE_VERSIONS:
            raise ValueError(f'`model_version` is "{model_version}" but should be one from {self._AVAILABLE_VERSIONS}')

        model_id = f'meta-llama/Llama-2-{model_version}b-chat-hf'
        super().__init__(model_id, quantization_config)


class Llama3(LlamaFamilyLLM):
    _AVAILABLE_VERSIONS: list = [8, 70]

    def __init__(
            self,
            model_version: int = 8,
            quantization_config: BitsAndBytesConfig = None,
    ):
        if model_version not in self._AVAILABLE_VERSIONS:
            raise ValueError(f'`model_version` is "{model_version}" but should be one from {self._AVAILABLE_VERSIONS}')

        model_id = f'meta-llama/Meta-Llama-3-{model_version}B-Instruct'
        super().__init__(model_id, quantization_config)


class Llama3dot1(LlamaFamilyLLM):
    _AVAILABLE_VERSIONS: list = [8, 70, 405]

    def __init__(
            self,
            model_version: int = 8,
            quantization_config: BitsAndBytesConfig = None,
    ):
        if model_version not in self._AVAILABLE_VERSIONS:
            raise ValueError(f'`model_version` is "{model_version}" but should be one from {self._AVAILABLE_VERSIONS}')

        model_id = f'meta-llama/Meta-Llama-3.1-{model_version}B-Instruct'
        super().__init__(model_id, quantization_config)


class Llama3dot3(LlamaFamilyLLM):
    _AVAILABLE_VERSIONS: list = [70]

    def __init__(
            self,
            model_version: int = 70,
            quantization_config: BitsAndBytesConfig = None,
    ):
        if model_version not in self._AVAILABLE_VERSIONS:
            raise ValueError(f'`model_version` is "{model_version}" but should be one from {self._AVAILABLE_VERSIONS}')

        model_id = f'meta-llama/Llama-3.3-{model_version}B-Instruct'
        super().__init__(model_id, quantization_config)


class CodeLlama(LlamaFamilyLLM):
    _AVAILABLE_VERSIONS: list = [7, 13, 34, 70]

    def __init__(
            self,
            model_version: int = 7,
            quantization_config: BitsAndBytesConfig = None,
    ):
        if model_version not in self._AVAILABLE_VERSIONS:
            raise ValueError(f'`model_version` is "{model_version}" but should be one from {self._AVAILABLE_VERSIONS}')

        model_id = f'meta-llama/CodeLlama-{model_version}b-Instruct-hf'
        super().__init__(model_id, quantization_config)
