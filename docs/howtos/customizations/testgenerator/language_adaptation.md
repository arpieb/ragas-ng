## Synthetic test generation from non-English corpus

In this guide, you'll learn how to generate synthetic test data in a target language. The sample corpus is a set of **French** Wikipedia articles about cities, and the goal is to generate test queries and answers in **Spanish**. The same steps work for any target language.

Without adaptation, the generated queries follow the language of the documents. Adapting the query synthesizer's prompts to Spanish makes them ask for Spanish output, whatever language the documents are in.

### Download and load the corpus


```python
! git clone https://huggingface.co/datasets/vibrantlabsai/Sample_non_english_corpus
```

Load each article as a ragas `Document`. No document-loader library is needed for plain text files.


```python
from pathlib import Path

from ragas.testset.document import Document

docs = [
    Document(
        page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name}
    )
    for path in sorted(Path("Sample_non_english_corpus").glob("*.txt"))
]
[doc.metadata["source"] for doc in docs]
```




    ['Londres.txt',
     'Madrid.txt',
     'New York.txt',
     'Pékin.txt',
     'Sydney.txt',
     'Tokyo.txt']



### Initialize the models

Any [`llm_factory()`](../../llm-adapters.md) LLM works. The outputs on this page were produced with `qwen3.8:27b` and `nomic-embed-text` on a local [Ollama](https://ollama.com) server, through its OpenAI-compatible API. To use OpenAI instead, swap in the commented lines.


```python
from openai import OpenAI

from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
generator_llm = llm_factory(
    "qwen3.8:27b", provider="openai", client=client, max_tokens=8192
)
generator_embeddings = embedding_factory(
    "openai", model="nomic-embed-text", client=client
)

# OpenAI:
# client = OpenAI()
# generator_llm = llm_factory("gpt-6-luna", client=client)
# generator_embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)
```

### Set up the persona and transforms
You can create personas automatically; see [Persona generation](./persona_generator.md). To keep this simple, this guide uses one predefined persona, two basic transforms and a simple query distribution.


```python
from ragas.testset.persona import Persona

personas = [
    Persona(
        name="curious student",
        role_description="A student who is curious about the world and wants to learn more about different cultures and languages",
    ),
]
```


```python
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter

# split the articles into chunks, then extract the named entities that
# single-hop queries are built around
transforms = [HeadlineSplitter(), NERExtractor(llm=generator_llm)]
```

### Initialize the test generator


```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(
    llm=generator_llm, embedding_model=generator_embeddings, persona_list=personas
)
```

### Adapt the queries to the target language

Load the query synthesizers and adapt their prompts to Spanish. Adaptation translates each prompt's examples, and with `adapt_instruction=True` its instruction too. The adapted query and answer prompt also tells the model to write in Spanish. That is what makes the output Spanish when the documents are French. The prompts that match personas and entities keep those values as they are, so the matching still works.

Adaptation uses the LLM, so use the best model you have, and save the adapted prompts with `save_prompts()` to reuse them.

--8<--
running_async.md
--8<--


```python
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
]

for query, _ in distribution:
    prompts = await query.adapt_prompts(
        "spanish", llm=generator_llm, adapt_instruction=True
    )
    query.set_prompts(**prompts)
```

### Generate


```python
dataset = generator.generate_with_docs(
    docs,
    testset_size=5,
    transforms=transforms,
    query_distribution=distribution,
)
```


```python
eval_dataset = dataset.to_dataset("spanish_testset")

for sample in eval_dataset:
    print("Query:", sample.user_input)
    print("Reference:", sample.reference)
    print()
```

    Query: ¿Cuál es la relación geográfica entre la ciudad de Londres y el estuario del río Tamise?
    Reference: Londres se encuentra situada cerca del estuario del río Tamise, en el sureste de Inglaterra.
    
    Query: que es el Reall Madid?
    Reference: Según el contexto, Madrid alberga dos de los más grandes clubes de fútbol del mundo, el Real Madrid y el Atlético de Madrid.
    
    Query: wall street es donde esta bolsa de new york?
    Reference: Wall Street es el barrio financiero de New York, anclado en el Lower Manhattan, que funciona como la «capital financiera del mundo». Alberga las dos bolsas más grandes del mundo por capitalización: el New York Stock Exchange (Bolsa de New York) y el NASDAQ. Además, el nuevo One World Trade Center, situado en esa zona, es el rascacielos más alto de América del Norte.
    
    Query: ¿Que fue lo que paso con Al-Qaida en New York el 11 de septiembre de 2001?
    Reference: El 11 de septiembre de 2001, New York fue golpeada por el atentado más grave que jamás haya afectado a los Estados Unidos: dos aviones de línea secuestrados por terroristas miembros de Al-Qaïda impactaron las torres gemelas del World Trade Center y las destruyeron por completo. En 2021, la reconstrucción del barrio aún no estaba terminada.
    
    Query: qué relación tiene Kubilai Khan con la ciudad de Pekín y bajo qué nombre la convirtió en su capital principal
    Reference: El príncipe mongol Kubilai Khan eligió a Pekín como su capital principal, dándole el nombre de Dadu, que significa «gran metrópolis».
    


The queries and answers are in Spanish, generated from the French articles. The queries vary in style on purpose: some are well-formed questions, some are web-search phrasings, and some have poor grammar or misspellings, as real users' queries do. Your output will differ from run to run.

That's it: you can customize the test generation process further as you need.
