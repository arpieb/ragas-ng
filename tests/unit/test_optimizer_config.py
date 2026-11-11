from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorLLM


def test_load_config():
    from ragas.config import DemonstrationConfig, InstructionConfig

    llm = MagicMock(spec=InstructorLLM)
    embedding = MagicMock(spec=BaseRagasEmbedding)
    inst_config = InstructionConfig(llm=llm)
    demo_config = DemonstrationConfig(embedding=embedding)
    assert inst_config.llm is llm
    assert demo_config.embedding is embedding


def test_configs_take_modern_models_only(fake_llm, fake_embedding):
    from ragas.config import DemonstrationConfig, InstructionConfig

    # the conftest fakes are legacy BaseRagasLLM / BaseRagasEmbeddings
    with pytest.raises(ValidationError, match="InstructorBaseRagasLLM"):
        InstructionConfig(llm=fake_llm)
    with pytest.raises(ValidationError, match="BaseRagasEmbedding"):
        DemonstrationConfig(embedding=fake_embedding)


def test_each_config_gets_its_own_optimizer():
    from ragas.config import InstructionConfig

    a = InstructionConfig(llm=MagicMock(spec=InstructorLLM))
    b = InstructionConfig(llm=MagicMock(spec=InstructorLLM))
    assert a.optimizer is not b.optimizer
