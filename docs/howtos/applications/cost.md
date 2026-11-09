# How to estimate Cost and Usage of evaluations and testset generation

--8<--
earlier_run_outputs.md
--8<--

When using LLMs for evaluation and test set generation, cost will be an important factor. Ragas provides you some tools to help you with that.

## Implement `TokenUsageParser`

By default Ragas does not calculate token usage for `evaluate()`, because providers report it in different shapes. To get usage data, pass a `TokenUsageParser`.

A `TokenUsageParser` is a function that takes the provider's raw completion and returns the `TokenUsage` that Ragas expects. Ragas ships parsers for OpenAI-shaped (`get_token_usage_for_openai`), Anthropic, Bedrock and Azure AI responses. LLMs created with `llm_factory()` call through LiteLLM, which returns OpenAI-shaped responses, so `get_token_usage_for_openai` works for them whatever the provider.

For example, here is the OpenAI parser applied to a completion:


```python
from openai import OpenAI

# lets import a parser for OpenAI
from ragas.cost import get_token_usage_for_openai

client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-6-luna",
    messages=[{"role": "user", "content": "hai there"}],
)

get_token_usage_for_openai(completion)
```
Output
```
TokenUsage(input_tokens=9, output_tokens=9, model='')
```


You can define your own or import parsers if they are defined. If you would like to suggest parser for LLM providers or contribute your own ones please check out this [issue](https://github.com/vibrantlabsai/ragas/issues/1151) 🙂.

## Token Usage for Evaluations

Let's use the `get_token_usage_for_openai` parser to calculate the token usage for an evaluation.


```python
from datasets import load_dataset

from ragas import EvaluationDataset

dataset = load_dataset("vibrantlabsai/amnesty_qa", "english_v3")

eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])
```
Output
```
Repo card metadata block was not found. Setting CardData to empty.
```

You can pass in the parser to the `evaluate()` function and the cost will be calculated and returned in the `Result` object.


```python
from ragas import evaluate
from ragas.cost import get_token_usage_for_openai
from ragas.llms import llm_factory
from ragas.metrics import LLMContextRecall

evaluator_llm = llm_factory("gpt-6-luna", client=client)

result = evaluate(
    eval_dataset,
    metrics=[LLMContextRecall()],
    llm=evaluator_llm,
    token_usage_parser=get_token_usage_for_openai,
)
```
Output
```
Evaluating:   0%|          | 0/20 [00:00<?, ?it/s]
```


```python
result.total_tokens()
```
Output
```
TokenUsage(input_tokens=25097, output_tokens=3757, model='')
```


You can compute the cost for each run by passing in the cost per token to `Result.total_cost()` function.

In this example the model costs $5 per 1M input tokens and $15 per 1M output tokens. Use your own model's prices.


```python
result.total_cost(cost_per_input_token=5 / 1e6, cost_per_output_token=15 / 1e6)
```

Output
```
1.1692900000000002
```


## Token Usage for Testset Generation

You can use the same parser for testset generation: pass `token_usage_parser` to `generate()`, `generate_with_docs()` or `generate_with_chunks()`. With the `generate_with_*` methods, the total includes building the knowledge graph (the transforms), which is usually most of the cost. `generate()` on an existing knowledge graph counts only generation, since the graph is already built. Only LLMs created with `llm_factory()` report usage.

For an example let's load an existing KnowledgeGraph and generate a testset. If you want to know more about how to generate a testset please check out the [testset generation](../../getstarted/rag_testset_generation.md#a-deeper-look).


```python
from ragas.testset.graph import KnowledgeGraph

# loading an existing KnowledgeGraph
# make sure to change the path to the location of the KnowledgeGraph file
kg = KnowledgeGraph.load("../../../experiments/scratchpad_kg.json")
kg
```

Output
```
KnowledgeGraph(nodes: 47, relationships: 109)
```


### Choose your LLM

--8<--
choose_generator_llm.md
--8<--


```python
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator

tg = TestsetGenerator(llm=llm_factory(), knowledge_graph=kg)
# generating a testset
testset = tg.generate(testset_size=10, token_usage_parser=get_token_usage_for_openai)
```


```python
# total cost for the generation process
testset.total_cost(cost_per_input_token=5 / 1e6, cost_per_output_token=15 / 1e6)
```

Output
```
0.20967000000000002
```

