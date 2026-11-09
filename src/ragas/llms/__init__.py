from ragas.llms.base import (
    DEFAULT_LLM_MODEL,
    BaseRagasLLM,
    InstructorBaseRagasLLM,
    InstructorLLM,
    InstructorTypeVar,
    LlamaIndexLLMWrapper as _LlamaIndexLLMWrapper,
    default_llm,
    llm_factory,
)
from ragas.llms.haystack_wrapper import HaystackLLMWrapper
from ragas.llms.litellm_llm import LiteLLMStructuredLLM
from ragas.llms.oci_genai_wrapper import OCIGenAIWrapper, oci_genai_factory
from ragas.utils import DeprecationHelper

# Create deprecation wrappers for legacy classes
LlamaIndexLLMWrapper = DeprecationHelper(
    _LlamaIndexLLMWrapper,
    "LlamaIndexLLMWrapper is deprecated and will be removed in a future version. "
    "Use llm_factory instead: "
    "from openai import OpenAI; "
    "from ragas.llms import llm_factory; "
    f"llm = llm_factory('{DEFAULT_LLM_MODEL}', client=OpenAI(api_key='...'))",
)

__all__ = [
    "BaseRagasLLM",
    "HaystackLLMWrapper",
    "InstructorBaseRagasLLM",
    "InstructorLLM",
    "LlamaIndexLLMWrapper",
    "LiteLLMStructuredLLM",
    "OCIGenAIWrapper",
    "InstructorTypeVar",
    "default_llm",
    "llm_factory",
    "oci_genai_factory",
]
