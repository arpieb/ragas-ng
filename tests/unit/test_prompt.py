import copy

import pytest
from pydantic import BaseModel

from ragas.prompt import StringIO

expected_generate_output_signature = """\
Please return the output in the following JSON format based on the StringIO model:
{
    "text": "str"
}\
"""


def test_process_fields():
    from enum import Enum

    from ragas.prompt import BasePrompt, StringIO

    class Categories(str, Enum):
        science = "science"
        commerce = "commerce"
        agriculture = "agriculture"
        economics = "economics"

    class InputModel(BaseModel):
        category: Categories

    class JokeGenerator(BasePrompt[InputModel, StringIO]):
        instruction = "Generate a joke in the category of {category}."
        output_model = StringIO

    p = JokeGenerator()
    _ = p._generate_output_signature()

    # assert expected_generate_output_signature == generation


@pytest.mark.asyncio
async def test_pydantic_prompt_io():
    from ragas.prompt import BasePrompt, StringIO

    class Prompt(BasePrompt[StringIO, StringIO]):
        instruction = ""
        input_model = StringIO
        output_model = StringIO

    p = Prompt()
    assert p.input_model == StringIO
    assert p.output_model == StringIO

    assert p._generate_examples() == ""


def test_pydantic_prompt_examples():
    from ragas.prompt import BasePrompt

    class Prompt(BasePrompt[StringIO, StringIO]):
        instruction = ""
        input_model = StringIO
        output_model = StringIO
        examples = [
            (StringIO(text="hello"), StringIO(text="hello")),
            (StringIO(text="world"), StringIO(text="world")),
        ]

    _ = Prompt()
    # assert p.generate_examples() == "hello -> hello\nworld -> world"


def test_prompt_hash():
    from ragas.prompt import BasePrompt, StringIO

    class Prompt(BasePrompt[StringIO, StringIO]):
        instruction = "You are a helpful assistant."
        input_model = StringIO
        output_model = StringIO

    p = Prompt()
    p_copy = Prompt()
    assert hash(p) == hash(p_copy)
    assert p == p_copy
    p.instruction = "You are a helpful assistant. And some more"
    assert hash(p) != hash(p_copy)
    assert p != p_copy


def test_prompt_hash_in_ragas():
    # check with a prompt inside ragas
    from ragas.testset.synthesizers.multi_hop import MultiHopAbstractQuerySynthesizer
    from tests.unit.testset_fakes import fake_llm

    synthesizer = MultiHopAbstractQuerySynthesizer(llm=fake_llm())
    prompts = synthesizer.get_prompts()
    for prompt in prompts.values():
        assert hash(prompt) == hash(prompt)
        assert prompt == prompt

    # change instruction and check if hash changes
    for prompt in prompts.values():
        old_prompt = copy.deepcopy(prompt)
        prompt.instruction = "You are a helpful assistant."
        assert hash(prompt) != hash(old_prompt)
        assert prompt != old_prompt


def test_prompt_save_load(tmp_path):
    from ragas.prompt import BasePrompt, StringIO

    class Prompt(BasePrompt[StringIO, StringIO]):
        instruction = "You are a helpful assistant."
        input_model = StringIO
        output_model = StringIO
        examples = [
            (StringIO(text="hello"), StringIO(text="hello")),
            (StringIO(text="world"), StringIO(text="world")),
        ]

    p = Prompt()
    file_path = tmp_path / "test_prompt.json"
    p.save(file_path)
    p1 = Prompt.load(file_path)
    assert hash(p) == hash(p1)
    assert p == p1


def test_prompt_save_load_language(tmp_path):
    from ragas.prompt import BasePrompt, StringIO

    class Prompt(BasePrompt[StringIO, StringIO]):
        instruction = "You are a helpful assistant."
        language = "spanish"
        input_model = StringIO
        output_model = StringIO
        examples = [
            (StringIO(text="hello"), StringIO(text="hello")),
            (StringIO(text="world"), StringIO(text="world")),
        ]

    p_spanish = Prompt()
    file_path = tmp_path / "test_prompt_spanish.json"
    p_spanish.save(file_path)
    p_spanish_loaded = Prompt.load(file_path)
    assert hash(p_spanish) == hash(p_spanish_loaded)
    assert p_spanish == p_spanish_loaded


def test_save_existing_prompt(tmp_path):
    from ragas.testset.synthesizers.prompts import ThemesPersonasMatchingPrompt

    p = ThemesPersonasMatchingPrompt()
    file_path = tmp_path / "test_prompt.json"
    p.save(file_path)
    p2 = ThemesPersonasMatchingPrompt.load(file_path)
    assert p == p2


def test_prompt_class_attributes():
    """
    We are using class attributes to store the prompt instruction and examples.
    We want to make sure there is no relationship between the class attributes
    and instance.
    """
    from ragas.testset.synthesizers.prompts import ThemesPersonasMatchingPrompt

    p = ThemesPersonasMatchingPrompt()
    p_another_instance = ThemesPersonasMatchingPrompt()
    assert p.instruction == p_another_instance.instruction
    assert p.examples == p_another_instance.examples
    p.instruction = "You are a helpful assistant."
    p.examples = []
    assert p.instruction != p_another_instance.instruction
    assert p.examples != p_another_instance.examples
