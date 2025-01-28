from mistral_common.protocol.instruct.messages import BaseMessage, SystemMessage, UserMessage, AssistantMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.models.mistral.modeling_mistral import MistralForCausalLM

from llms.abstract import InstructedLLM


class MistralFamilyLLM(InstructedLLM):
    """An InstructedLLM implementation from the `mistral` models."""
    _model_id: str
    _tokenizer: MistralTokenizer
    _model: MistralForCausalLM
    _base_terminators: list[int]
    _eos_id: int
    _pad_id: int

    def __init__(self, model_id: str, quantization_config: BitsAndBytesConfig = None):
        self._model_id = model_id

        self._tokenizer = MistralTokenizer.v3()
        self._eos_id = self._tokenizer.instruct_tokenizer.tokenizer.eos_id
        self._base_terminators = [self._eos_id]
        # SEE https://docs.mistral.ai/guides/tokenization/#control-tokens
        assert self._eos_id == self._tokenizer.instruct_tokenizer.tokenizer.get_control_token('</s>')
        self._pad_id = self._tokenizer.instruct_tokenizer.tokenizer.get_control_token('<pad>')

        if quantization_config is None:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        self._model = AutoModelForCausalLM.from_pretrained(self._model_id, quantization_config=quantization_config)

    def chat(
            self,
            messages: list[dict],
            max_new_tokens: int = 256,
            temperature: float = 0.6,
            top_p: float = 0.9,
            terminators: list[str] = None,
            **_,
    ) -> str:
        if not terminators:
            terminators = list()
        else:
            terminators = [self._tokenizer.instruct_tokenizer.tokenizer.get_control_token(c) for c in terminators]

        completion_request = ChatCompletionRequest(messages=self.__from_dict_to_message(messages))

        input_ids = torch.tensor(
            [self._tokenizer.encode_chat_completion(completion_request).tokens], dtype=torch.int32
        ).to(self._model.device)

        output_ids = self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=[*self._base_terminators, *terminators],
            pad_token_id=self._pad_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response_ids = output_ids[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(response_ids.tolist())

    @staticmethod
    def __from_dict_to_message(dict_messages: list[dict]) -> list[BaseMessage]:
        conversion_dict: dict[str, type[BaseMessage]] = {
            'system': SystemMessage,
            'user': UserMessage,
            'assistant': AssistantMessage,
        }

        messages: list[BaseMessage] = list()
        for dm in dict_messages:
            role, content = dm.values()
            msg_class = conversion_dict[role]
            messages.append(msg_class(content=content))

        return messages
