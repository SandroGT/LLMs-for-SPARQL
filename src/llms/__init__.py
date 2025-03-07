"""The `llms` package provides an extensible framework for integrating and managing various Large Language Models (LLMs)
to handle diverse tasks. It is designed to offer a standardized, easy-to-use, and customizable interface for working
with LLMs in current experiments, without relying on more complex and feature-rich frameworks.

## Key Components

1. **Abstract base classes:**
   - `InstructedLLM`: defines the interface for causal language models fine-tuned to follow instructions. It requires
     concrete implementations to provide methods for interacting with the model (`chat`) and executing structured
     tasks (`task`).
   - `AgentLLM`: offers a higher-level abstraction for task-specific agents built on top of an `InstructedLLM`. It
     provides mechanisms for formatting system and user prompts, parsing model responses, and running tasks seamlessly.

2. **Concrete implementations:**
   - Implementations for popular LLMs such as GPT, Llama, or Mistral, extending `InstructedLLM` to adapt each model's
     API and behavior to the standardized interface.
"""
from llms.abstract import AgentLLM, InstructedLLM
from llms.gpt import GPTFamilyLLM
from llms.llama import LlamaFamilyLLM, Llama2, Llama3, Llama3dot1, Llama3dot3, CodeLlama
from llms.mistral import MistralFamilyLLM
from llms.ollama_server import OllamaServerLLM
from llms.transformers import TransformersLLM
