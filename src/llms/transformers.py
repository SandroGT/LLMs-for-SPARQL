import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel, BitsAndBytesConfig

from llms.abstract import InstructedLLM


class TransformersLLM(InstructedLLM):
    """An InstructedLLM implementation from the `transformers` library."""
    _model_id: str
    _tokenizer: PreTrainedTokenizer
    _model: PreTrainedModel
    _base_terminators: list[int]

    def __init__(self, model_id: str, quantization_config: BitsAndBytesConfig = None):
        if 'HF_TOKEN' not in os.environ:
            raise ValueError('HF_TOKEN environment variable is not set')

        self._model_id = model_id

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._base_terminators = [self._tokenizer.eos_token_id]

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
            terminators = [self._tokenizer.convert_tokens_to_ids(c) for c in terminators]

        input_ids = torch.Tensor(self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt'
        )).to(self._model.device)

        output_ids = self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=[*self._base_terminators, *terminators],
            pad_token_id=self._tokenizer.pad_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response_ids = output_ids[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(response_ids, skip_special_tokens=True)
