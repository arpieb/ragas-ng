"""Collections metrics explain their scores, and metrics_experiment keeps it (#116)."""

from __future__ import annotations

import pickle
import re
import typing as t
from unittest.mock import MagicMock

import pytest

from ragas import Dataset, SingleTurnSample, metrics_experiment
from ragas.backends import InMemoryBackend
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorLLM
from ragas.messages import AIMessage, HumanMessage
from ragas.metrics import collections as C
from ragas.metrics.collections._reasons import count_phrase
from ragas.metrics.collections.agent_goal_accuracy import util as ag
from ragas.metrics.collections.answer_correctness import util as ac
from ragas.metrics.collections.answer_relevancy import util as ar
from ragas.metrics.collections.context_entity_recall import util as cer
from ragas.metrics.collections.context_precision import util as cp
from ragas.metrics.collections.context_recall import util as cr
from ragas.metrics.collections.faithfulness import util as fa
from ragas.metrics.collections.summary_score import util as ss
from ragas.metrics.collections.topic_adherence import util as ta
from ragas.metrics.result import ScoreWithReason


def scripted(handlers: t.Dict[type, t.Callable[[str], t.Any]]):
    llm = MagicMock(spec=InstructorLLM)

    async def agenerate(prompt, model, **kwargs):
        return handlers[model](prompt)

    llm.agenerate = agenerate
    return llm


def faithfulness_llm():
    return scripted(
        {
            fa.StatementGeneratorOutput: lambda p: fa.StatementGeneratorOutput(
                statements=["Paris is in France.", "Paris has 90 million people."]
            ),
            fa.NLIStatementOutput: lambda p: fa.NLIStatementOutput(
                statements=[
                    fa.StatementFaithfulnessAnswer(
                        statement="Paris is in France.", reason="stated", verdict=1
                    ),
                    fa.StatementFaithfulnessAnswer(
                        statement="Paris has 90 million people.",
                        reason="population not mentioned",
                        verdict=0,
                    ),
                ]
            ),
        }
    )


# --- plumbing -----------------------------------------------------------------


def test_score_with_reason_is_a_float():
    s = ScoreWithReason(0.5, "because")
    assert s == 0.5 and s + 1 == 1.5 and isinstance(s, float)
    assert s.reason == "because"
    back = pickle.loads(pickle.dumps(s))
    assert back == 0.5 and back.reason == "because"


def test_metrics_experiment_keeps_collections_reasons():
    dataset = Dataset(
        "rows",
        InMemoryBackend(),
        data_model=SingleTurnSample,
        data=[
            SingleTurnSample(
                user_input="q", response="r", retrieved_contexts=["Paris is in France."]
            )
        ],
    )
    exp = metrics_experiment([C.Faithfulness(llm=faithfulness_llm())]).run(dataset)
    (row,) = list(exp)
    assert row["faithfulness"] == 0.5
    assert type(row["faithfulness"]) is float  # scores stay plain numbers
    assert "1 of 2 statements" in row["faithfulness_reason"]


# --- per-metric explanations ---------------------------------------------------


@pytest.mark.asyncio
async def test_faithfulness_reason_lists_each_statement():
    r = await C.Faithfulness(llm=faithfulness_llm()).ascore(
        user_input="q", response="r", retrieved_contexts=["c"]
    )
    assert r.value == 0.5
    assert r.reason.splitlines() == [
        "1 of 2 statements in the response is supported by the retrieved context.",
        "- [supported] Paris is in France.: stated",
        "- [not supported] Paris has 90 million people.: population not mentioned",
    ]


@pytest.mark.asyncio
async def test_factual_correctness_reason_covers_both_directions():
    def nli(prompt):
        tail = prompt.split("Now perform")[-1]
        claims = re.findall(r'"((?:resp|ref) claim \d)"', tail)
        return fa.NLIStatementOutput(
            statements=[
                fa.StatementFaithfulnessAnswer(
                    statement=c, reason=f"why {c}", verdict=1
                )
                for c in claims
            ]
        )

    from ragas.metrics.collections.factual_correctness import util as fc

    def decompose(prompt):
        who = "resp" if "RESPONSE" in prompt.split("Now perform")[-1] else "ref"
        return fc.ClaimDecompositionOutput(claims=[f"{who} claim 0", f"{who} claim 1"])

    llm = scripted({fc.ClaimDecompositionOutput: decompose, fa.NLIStatementOutput: nli})
    r = await C.FactualCorrectness(llm=llm).ascore(response="RESPONSE", reference="REF")
    assert r.value == 1.0
    assert "Precision: 2 of 2 claims in the response are supported" in r.reason
    assert "Recall: 2 of 2 claims in the reference are supported" in r.reason
    assert "- [supported] ref claim 1: why ref claim 1" in r.reason


@pytest.mark.asyncio
async def test_context_recall_reason():
    def judge(prompt):
        return cr.ContextRecallOutput(
            classifications=[
                cr.ContextRecallClassification(
                    statement="A is B.", reason="in ctx", attributed=1
                ),
                cr.ContextRecallClassification(
                    statement="C is D.", reason="absent", attributed=0
                ),
            ]
        )

    r = await C.ContextRecall(llm=scripted({cr.ContextRecallOutput: judge})).ascore(
        user_input="q", retrieved_contexts=["ctx"], reference="A is B. C is D."
    )
    assert r.value == 0.5
    assert r.reason.splitlines()[0] == (
        "1 of 2 sentences of the reference is attributable to the retrieved context."
    )
    assert "- [not attributed] C is D.: absent" in r.reason


@pytest.mark.asyncio
async def test_context_precision_reason_is_per_rank():
    verdicts = iter([0, 1])
    llm = scripted(
        {
            cp.ContextPrecisionOutput: lambda p: cp.ContextPrecisionOutput(
                reason="judged", verdict=next(verdicts)
            )
        }
    )
    r = await C.ContextPrecision(llm=llm).ascore(
        user_input="q", reference="ref", retrieved_contexts=["first ctx", "second ctx"]
    )
    assert r.value == pytest.approx(0.5)
    assert "- [not useful] #1 'first ctx': judged" in r.reason
    assert "- [useful] #2 'second ctx': judged" in r.reason


@pytest.mark.asyncio
async def test_answer_relevancy_reason_lists_generated_questions():
    llm = scripted(
        {
            ar.AnswerRelevanceOutput: lambda p: ar.AnswerRelevanceOutput(
                question="Where is Paris?", noncommittal=0
            )
        }
    )
    emb = MagicMock(spec=BaseRagasEmbedding)

    async def one(text, **k):
        return [1.0, 0.0]

    async def many(texts, **k):
        return [[1.0, 0.0] for _ in texts]

    emb.aembed_text, emb.aembed_texts = one, many
    r = await C.AnswerRelevancy(llm=llm, embeddings=emb, strictness=1).ascore(
        user_input="Where is Paris?", response="In France."
    )
    assert "Mean similarity 1.00" in r.reason
    assert "- 'Where is Paris?' (similarity 1.00)" in r.reason


@pytest.mark.asyncio
async def test_answer_correctness_reason_lists_tp_fp_fn():
    s = lambda text: ac.StatementsWithReason(statement=text, reason=f"r-{text}")  # noqa: E731
    llm = scripted(
        {
            fa.StatementGeneratorOutput: lambda p: fa.StatementGeneratorOutput(
                statements=["x"]
            ),
            ac.ClassificationWithReason: lambda p: ac.ClassificationWithReason(
                TP=[s("tp")], FP=[s("fp")], FN=[]
            ),
        }
    )
    r = await C.AnswerCorrectness(llm=llm, weights=[1.0, 0.0]).ascore(
        user_input="q", response="r", reference="ref"
    )
    assert "Factuality 0.67" in r.reason
    assert "- tp: r-tp" in r.reason and "- fp: r-fp" in r.reason
    assert "Missing from the response" not in r.reason  # empty FN section omitted


@pytest.mark.asyncio
async def test_context_entity_recall_reason():
    def ents(prompt):
        tail = prompt.split("Now perform")[-1]
        if "REFERENCE" in tail:
            return cer.EntitiesList(entities=["Paris", "1889"])
        return cer.EntitiesList(entities=["Paris"])

    r = await C.ContextEntityRecall(llm=scripted({cer.EntitiesList: ents})).ascore(
        reference="REFERENCE", retrieved_contexts=["ctx"]
    )
    assert "1 of 2 entities in the reference" in r.reason
    assert "- [missing] 1889" in r.reason


@pytest.mark.asyncio
async def test_agent_goal_accuracy_surfaces_the_judge_reason():
    llm = scripted(
        {
            ag.WorkflowOutput: lambda p: ag.WorkflowOutput(
                user_goal="book a table", end_state="table booked"
            ),
            ag.CompareOutcomeOutput: lambda p: ag.CompareOutcomeOutput(
                reason="the table was booked as asked", verdict="1"
            ),
        }
    )
    r = await C.AgentGoalAccuracyWithoutReference(llm=llm).ascore(
        user_input=[HumanMessage(content="book"), AIMessage(content="booked")]
    )
    assert r.value == 1.0
    assert r.reason.startswith("Goal achieved. the table was booked as asked")


@pytest.mark.asyncio
async def test_summary_score_reason_lists_questions():
    llm = scripted(
        {
            ss.ExtractedKeyphrases: lambda p: ss.ExtractedKeyphrases(keyphrases=["k"]),
            ss.QuestionsGenerated: lambda p: ss.QuestionsGenerated(
                questions=["Q one?", "Q two?"]
            ),
            ss.AnswersGenerated: lambda p: ss.AnswersGenerated(answers=["1", "0"]),
        }
    )
    r = await C.SummaryScore(llm=llm, length_penalty=False).ascore(
        reference_contexts=["source"], response="summary"
    )
    assert "answers 1 of 2 questions" in r.reason
    assert "- [not answered] Q two?" in r.reason


@pytest.mark.asyncio
async def test_topic_adherence_reason():
    llm = scripted(
        {
            ta.TopicExtractionOutput: lambda p: ta.TopicExtractionOutput(
                topics=["weather", "stocks"]
            ),
            ta.TopicRefusedOutput: lambda p: ta.TopicRefusedOutput(
                refused_to_answer="stocks" in p.split("Now perform")[-1]
            ),
            ta.TopicClassificationOutput: lambda p: ta.TopicClassificationOutput(
                classifications=[True, False]
            ),
        }
    )
    r = await C.TopicAdherence(llm=llm).ascore(
        user_input=[HumanMessage(content="hi"), AIMessage(content="hello")],
        reference_topics=["weather"],
    )
    assert "- weather: answered, in scope" in r.reason
    assert "- stocks: refused, out of scope" in r.reason


@pytest.mark.parametrize(
    "k, n, noun, expected",
    [
        (1, 1, "entity", "1 of 1 entity"),
        (0, 2, "entity", "0 of 2 entities"),
        (2, 3, "statement", "2 of 3 statements"),
        (1, 2, "day", "1 of 2 days"),
    ],
)
def test_count_phrase(k, n, noun, expected):
    assert count_phrase(k, n, noun) == expected


@pytest.mark.parametrize(
    "k, n, expected",
    [
        (1, 1, "is"),  # 1 of 1 statement is
        (0, 1, "is"),  # 0 of 1 statement is
        (1, 3, "is"),  # 1 of 3 statements is
        (0, 2, "are"),  # 0 of 2 statements are
        (2, 3, "are"),  # 2 of 3 statements are
    ],
)
def test_count_verb_agrees_with_the_count(k, n, expected):
    # #200: the reasons used to say "1 of 1 statement ... are"
    from ragas.metrics.collections._reasons import count_verb

    assert count_verb(k, n) == expected
