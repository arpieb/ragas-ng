# LlamaIndex

--8<--
earlier_run_outputs.md
--8<--

[LlamaIndex](https://github.com/run-llama/llama_index) is a data framework for LLM applications to ingest, structure, and access private or domain-specific data. Makes it super easy to connect LLMs with your own data. But in order to figure out the best configuration for LlamaIndex and your data you need an object measure of the performance. This is where ragas comes in. Ragas will help you evaluate your `QueryEngine` and gives you the confidence to tweak the configuration to get the highest score.

This guide assumes you are familiar with the LlamaIndex framework.

## Building the Testset

You will need a testset to evaluate your `QueryEngine` against. You can either build one yourself or use the [Testset Generator Module](./../../getstarted/rag_testset_generation.md) in Ragas to get started with a small synthetic one.

Let's see how that works with LlamaIndex

## load the documents


```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./nyc_wikipedia").load_data()
```

Now let's initialize the `TestsetGenerator`. It takes an `llm_factory()` LLM
and a ragas embedding model; LiteLLM reaches the same providers LlamaIndex
does, so no LlamaIndex LLM wrapper is needed.


```python
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator

# generator with openai models, credentials from OPENAI_API_KEY
generator_llm = llm_factory("gpt-6-luna")
embeddings = embedding_factory("openai", model="text-embedding-3-large")

generator = TestsetGenerator(llm=generator_llm, embedding_model=embeddings)
```

Now you are all set to generate the dataset. A LlamaIndex `Document` keeps its
text in `text` rather than `page_content`, so pass a `document_adapter` that
converts each one to a ragas `Document`. Documents with no text are skipped.


```python
from ragas.testset.document import Document


def from_llama_index(doc):
    return Document(page_content=doc.text, metadata=doc.metadata)


# generate testset
testset = generator.generate_with_docs(
    documents,
    testset_size=5,
    document_adapter=from_llama_index,
)
```

The same adapter works for `generate_with_chunks()` if your LlamaIndex nodes
are already chunked, and the same pattern works for documents from any other
framework: return an object with `page_content` and `metadata`.


```python
df = testset.to_pandas()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>reference_contexts</th>
      <th>reference</th>
      <th>synthesizer_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cud yu pleese explane the role of New York Cit...</td>
      <td>[New York, often called New York City or NYC, ...</td>
      <td>New York City serves as the geographical and d...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>So like, what was New York City called before ...</td>
      <td>[History == === Early history === In the pre-C...</td>
      <td>Before it was called New York, the area was kn...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>what happen in new york with slavery and how i...</td>
      <td>[and rechristened it "New Orange" after Willia...</td>
      <td>In the early 18th century, New York became a c...</td>
      <td>single_hop_specifc_query_synthesizer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What historical significance does Long Island ...</td>
      <td>[&lt;1-hop&gt;\n\nHistory == === Early history === I...</td>
      <td>Long Island holds historical significance in t...</td>
      <td>multi_hop_specific_query_synthesizer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What role does the Staten Island Ferry play in...</td>
      <td>[&lt;1-hop&gt;\n\nto start service in 2017; this wou...</td>
      <td>The Staten Island Ferry plays a significant ro...</td>
      <td>multi_hop_specific_query_synthesizer</td>
    </tr>
  </tbody>
</table>
</div>



with a test dataset to test our `QueryEngine` lets now build one and evaluate it.

## Building the `QueryEngine`

To start lets build a `VectorStoreIndex` over the New York Cities' [Wikipedia page](https://en.wikipedia.org/wiki/New_York_City) as an example and use ragas to evaluate it.

Since we already loaded the dataset into `documents` lets use that.


```python
# build query engine
from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex.from_documents(documents)

query_engine = vector_index.as_query_engine()
```

Let's try a sample question from the generated testset to see if it is working


```python
# convert it to pandas dataset
df = testset.to_pandas()
df["user_input"][0]
```




    'Cud yu pleese explane the role of New York City within the Northeast megalopolis, and how it contributes to the cultural and economic vibrancy of the region?'




```python
response_vector = query_engine.query(df["user_input"][0])

print(response_vector)
```

    New York City serves as a key hub within the Northeast megalopolis, playing a significant role in enhancing the cultural and economic vibrancy of the region. Its status as a global center of creativity, entrepreneurship, and cultural diversity contributes to the overall dynamism of the area. The city's renowned arts scene, including Broadway theatre and numerous cultural institutions, attracts artists and audiences from around the world, enriching the cultural landscape of the Northeast megalopolis. Economically, New York City's position as a leading financial and fintech center, home to major stock exchanges and a bustling real estate market, bolsters the region's economic strength and influence. Additionally, the city's diverse culinary scene, influenced by its immigrant history, adds to the cultural richness of the region, making New York City a vital component of the Northeast megalopolis's cultural and economic tapestry.


## Evaluating the `QueryEngine`

Now that we have a `QueryEngine` for the `VectorStoreIndex`, we can evaluate it with Ragas.

In order to run an evaluation with Ragas and LlamaIndex you need 3 things

1. LlamaIndex `QueryEngine`: what we will be evaluating
2. Metrics: Ragas defines a set of metrics that can measure different aspects of the `QueryEngine`. The available metrics and their meaning can be found [here](../../concepts/metrics/available_metrics/index.md)
3. Questions: A list of questions that ragas will test the `QueryEngine` against.

first let's generate the questions. Ideally you should use that you see in production so that the distribution of question with which we evaluate matches the distribution of questions seen in production. This ensures that the scores reflect the performance seen in production but to start off we'll be using a few example questions.

Now let's set up the metrics we will be using to evaluate. Ragas metrics take an `llm_factory()` LLM (and, for `AnswerRelevancy`, ragas embeddings) when you create them. LiteLLM reaches the same providers LlamaIndex does, so you don't need to wrap a LlamaIndex LLM.


```python
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

evaluator_llm = llm_factory("gpt-6-luna")
metrics = [
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings),
    ContextPrecision(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
]
```

The testset holds the questions (`user_input`) and the expected answers (`reference`). Convert it to a `Dataset`:


```python
dataset = testset.to_dataset("nyc_questions")
len(dataset)
```




    6



The metrics also need what the `QueryEngine` returns for each question: the answer (`response`) and the text of the nodes it retrieved (`retrieved_contexts`). `evaluate_query_engine()` runs the engine over the dataset, fills them in, and scores the rows with [`metrics_experiment`](../../concepts/experimentation.md#scoring-a-dataset-in-one-call-metrics_experiment):


```python
from ragas import summarize
from ragas.integrations.llama_index import evaluate_query_engine

exp = evaluate_query_engine(query_engine, dataset, metrics, name="nyc_wiki")
```

In async code, `await aevaluate_query_engine(...)` instead. The dataset isn't modified; the returned `Experiment` holds the filled-in rows and a column per metric.


```python
# final scores
summarize(exp)
```

    {'faithfulness': {'mean': 0.7454, 'count': 6, 'missing': 0}, 'answer_relevancy': {'mean': 0.9348, 'count': 6, 'missing': 0}, 'context_precision': {'mean': 0.6667, 'count': 6, 'missing': 0}, 'context_recall': {'mean': 0.4667, 'count': 6, 'missing': 0}}


You can convert into a pandas DataFrame to run more analysis on it.


```python
exp.to_pandas()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>reference_contexts</th>
      <th>response</th>
      <th>reference</th>
      <th>faithfulness</th>
      <th>answer_relevancy</th>
      <th>context_precision</th>
      <th>context_recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cud yu pleese explane the role of New York Cit...</td>
      <td>[and its ideals of liberty and peace. In the 2...</td>
      <td>[New York, often called New York City or NYC, ...</td>
      <td>New York City plays a significant role within ...</td>
      <td>New York City serves as the geographical and d...</td>
      <td>0.615385</td>
      <td>0.918217</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>So like, what was New York City called before ...</td>
      <td>[New York City is the headquarters of the glob...</td>
      <td>[History == === Early history === In the pre-C...</td>
      <td>New York City was named New Amsterdam before i...</td>
      <td>Before it was called New York, the area was kn...</td>
      <td>1.000000</td>
      <td>0.967821</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>what happen in new york with slavery and how i...</td>
      <td>[=== Province of New York and slavery ===\n\nI...</td>
      <td>[and rechristened it "New Orange" after Willia...</td>
      <td>Slavery became a significant part of New York'...</td>
      <td>In the early 18th century, New York became a c...</td>
      <td>1.000000</td>
      <td>0.919264</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What historical significance does Long Island ...</td>
      <td>[==== River crossings ====\n\nNew York City is...</td>
      <td>[&lt;1-hop&gt;\n\nHistory == === Early history === I...</td>
      <td>Long Island played a significant role in the e...</td>
      <td>Long Island holds historical significance in t...</td>
      <td>0.500000</td>
      <td>0.931895</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What role does the Staten Island Ferry play in...</td>
      <td>[==== Buses ====\n\nNew York City's public bus...</td>
      <td>[&lt;1-hop&gt;\n\nto start service in 2017; this wou...</td>
      <td>The Staten Island Ferry serves as a vital mode...</td>
      <td>The Staten Island Ferry plays a significant ro...</td>
      <td>0.500000</td>
      <td>0.936920</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>How does Central Park's role as a cultural and...</td>
      <td>[==== State parks ====\n\nThere are seven stat...</td>
      <td>[&lt;1-hop&gt;\n\nCity has over 28,000 acres (110 km...</td>
      <td>Central Park's role as a cultural and historic...</td>
      <td>Central Park, located in middle-upper Manhatta...</td>
      <td>0.857143</td>
      <td>0.934841</td>
      <td>1.0</td>
      <td>0.8</td>
    </tr>
  </tbody>
</table>
</div>


