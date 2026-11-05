from __future__ import annotations

import asyncio
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks


class ResponseRelevanceOutput(BaseModel):
    question: str
    noncommittal: int


class ResponseRelevanceInput(BaseModel):
    response: str


class ResponseRelevancePrompt(
    PydanticPrompt[ResponseRelevanceInput, ResponseRelevanceOutput]
):
    instruction = """Generate a question for the given answer and Identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers"""
    input_model = ResponseRelevanceInput
    output_model = ResponseRelevanceOutput
    examples = [
        (
            ResponseRelevanceInput(
                response="""Albert Einstein was born in Germany.""",
            ),
            ResponseRelevanceOutput(
                question="Where was Albert Einstein born?",
                noncommittal=0,
            ),
        ),
        (
            ResponseRelevanceInput(
                response="""I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. """,
            ),
            ResponseRelevanceOutput(
                question="What was the groundbreaking feature of the smartphone invented in 2023?",
                noncommittal=1,
            ),
        ),
    ]


@dataclass
class ResponseRelevancy(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Attributes
    ----------
    name: string
        The name of the metrics
    strictness: int
        Here indicates the number questions generated per answer.
        Ideal range between 3 to 5.
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
            }
        }
    )
    output_type = MetricOutputType.CONTINUOUS

    question_generation: PydanticPrompt = ResponseRelevancePrompt()
    strictness: int = 3

    @staticmethod
    def _cosine(question_vec: t.Any, gen_vecs: t.Any, n: int) -> np.ndarray:
        q = np.asarray(question_vec).reshape(1, -1)
        g = np.asarray(gen_vecs).reshape(n, -1)
        norm = np.linalg.norm(g, axis=1) * np.linalg.norm(q, axis=1)
        return np.dot(g, q.T).reshape(-1) / norm

    def calculate_similarity(self, question: str, generated_questions: list[str]):
        """Synchronous similarity (kept for direct callers). Scoring uses
        ``acalculate_similarity``, which doesn't block the event loop."""
        assert self.embeddings is not None, (
            f"Error: '{self.name}' requires embeddings to be set."
        )
        return self._cosine(
            self.embeddings.embed_query(question),  # type: ignore[attr-defined]
            self.embeddings.embed_documents(generated_questions),  # type: ignore[attr-defined]
            len(generated_questions),
        )

    async def acalculate_similarity(
        self, question: str, generated_questions: list[str]
    ) -> np.ndarray:
        """Cosine similarity via the embeddings' async API.

        The sync ``embed_query``/``embed_documents`` calls used to run inside
        ``_ascore``, blocking the whole event loop for every embedding request
        and stalling unrelated concurrent jobs under ``evaluate()`` (#75).
        """
        emb = self.embeddings
        assert emb is not None, f"Error: '{self.name}' requires embeddings to be set."
        if hasattr(emb, "aembed_query") and hasattr(emb, "aembed_documents"):
            q = await emb.aembed_query(question)  # type: ignore[union-attr]
            g = await emb.aembed_documents(generated_questions)  # type: ignore[union-attr]
        elif hasattr(emb, "aembed_text") and hasattr(emb, "aembed_texts"):
            q = await emb.aembed_text(question)  # type: ignore[union-attr]
            g = await emb.aembed_texts(generated_questions)  # type: ignore[union-attr]
        else:
            q = await asyncio.to_thread(emb.embed_query, question)  # type: ignore[union-attr]
            g = await asyncio.to_thread(emb.embed_documents, generated_questions)  # type: ignore[union-attr]
        return self._cosine(q, g, len(generated_questions))

    def _questions_or_nan(
        self, answers: t.Sequence[ResponseRelevanceOutput]
    ) -> t.Optional[t.List[str]]:
        gen_questions = [answer.question for answer in answers]
        if all(q == "" for q in gen_questions):
            logger.warning(
                "Invalid JSON response. Expected dictionary with key 'question'"
            )
            return None
        return gen_questions

    def _calculate_score(
        self, answers: t.Sequence[ResponseRelevanceOutput], row: t.Dict
    ) -> float:
        gen_questions = self._questions_or_nan(answers)
        if gen_questions is None:
            return np.nan
        all_noncommittal = np.all([answer.noncommittal for answer in answers])
        cosine_sim = self.calculate_similarity(row["user_input"], gen_questions)
        return cosine_sim.mean() * int(not all_noncommittal)

    async def _acalculate_score(
        self, answers: t.Sequence[ResponseRelevanceOutput], row: t.Dict
    ) -> float:
        gen_questions = self._questions_or_nan(answers)
        if gen_questions is None:
            return np.nan
        all_noncommittal = np.all([answer.noncommittal for answer in answers])
        cosine_sim = await self.acalculate_similarity(row["user_input"], gen_questions)
        return cosine_sim.mean() * int(not all_noncommittal)

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt_input = ResponseRelevanceInput(response=row["response"])

        responses = await self.question_generation.generate_multiple(
            data=prompt_input, llm=self.llm, callbacks=callbacks, n=self.strictness
        )

        return await self._acalculate_score(responses, row)


class AnswerRelevancy(ResponseRelevancy):
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await super()._ascore(row, callbacks)


answer_relevancy = AnswerRelevancy()
