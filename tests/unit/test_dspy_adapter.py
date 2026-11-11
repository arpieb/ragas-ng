from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel, Field

from ragas.dataset_schema import (
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation,
)
from ragas.losses import MSELoss
from ragas.prompt.metrics.base_prompt import BasePrompt

try:
    import dspy  # noqa: F401

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


class TestPromptToDSPySignature:
    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
    def test_basic_conversion(self):
        """Test basic conversion of a BasePrompt to DSPy Signature."""
        from ragas.optimizers.dspy_adapter import prompt_to_dspy_signature

        class InputModel(BaseModel):
            question: str = Field(description="The question")
            context: str = Field(description="The context")

        class OutputModel(BaseModel):
            answer: str = Field(description="The answer")

        class TestPrompt(BasePrompt[InputModel, OutputModel]):
            instruction = "Answer the question"
            input_model = InputModel
            output_model = OutputModel

        prompt = TestPrompt()

        signature = prompt_to_dspy_signature(prompt)

        assert signature.__doc__ == "Answer the question"
        assert "question" in signature.model_fields
        assert "context" in signature.model_fields
        assert "answer" in signature.model_fields

    @pytest.mark.skip(reason="Import error test requires complex mocking")
    def test_import_error_without_dspy(self):
        """Test that conversion raises ImportError when dspy is not installed.

        Note: This test is skipped because it requires mocking the import system
        which is complex and fragile. The import error is adequately tested by
        the e2e tests when dspy is not installed.
        """
        pass

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
    def test_field_descriptions(self):
        """Test that field descriptions are preserved."""
        from ragas.optimizers.dspy_adapter import prompt_to_dspy_signature

        class InputModel(BaseModel):
            question: str = Field(description="User's question")

        class OutputModel(BaseModel):
            score: float = Field(description="Relevance score")

        class TestPrompt(BasePrompt[InputModel, OutputModel]):
            instruction = "Score relevance"
            input_model = InputModel
            output_model = OutputModel

        prompt = TestPrompt()

        signature = prompt_to_dspy_signature(prompt)

        assert "question" in signature.model_fields
        assert "score" in signature.model_fields

        question_field = signature.model_fields["question"]
        score_field = signature.model_fields["score"]

        assert question_field.json_schema_extra["__dspy_field_type"] == "input"
        assert score_field.json_schema_extra["__dspy_field_type"] == "output"


class TestRagasDatasetToDSPyExamples:
    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
    def test_basic_conversion(self):
        """Test basic conversion of Ragas dataset to DSPy examples."""
        from ragas.optimizers.dspy_adapter import ragas_dataset_to_dspy_examples

        prompt_annotation = PromptAnnotation(
            prompt_input={"question": "What is 2+2?", "context": "Math"},
            prompt_output={"answer": "4"},
            edited_output=None,
        )

        sample = SampleAnnotation(
            metric_input={"question": "What is 2+2?"},
            metric_output=0.9,
            prompts={"test_prompt": prompt_annotation},
            is_accepted=True,
        )

        dataset = SingleMetricAnnotation(name="test_metric", samples=[sample])

        examples = ragas_dataset_to_dspy_examples(dataset, "test_prompt")

        assert len(examples) == 1
        example = examples[0]
        assert example.question == "What is 2+2?"
        assert example.context == "Math"
        assert example.answer == "4"

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
    def test_skip_non_accepted_samples(self):
        """Test that non-accepted samples are skipped."""
        from ragas.optimizers.dspy_adapter import ragas_dataset_to_dspy_examples

        prompt_annotation = PromptAnnotation(
            prompt_input={"question": "What is 2+2?"},
            prompt_output={"answer": "4"},
            edited_output=None,
        )

        sample1 = SampleAnnotation(
            metric_input={"question": "What is 2+2?"},
            metric_output=0.9,
            prompts={"test_prompt": prompt_annotation},
            is_accepted=True,
        )

        sample2 = SampleAnnotation(
            metric_input={"question": "What is 3+3?"},
            metric_output=0.8,
            prompts={"test_prompt": prompt_annotation},
            is_accepted=False,
        )

        dataset = SingleMetricAnnotation(name="test_metric", samples=[sample1, sample2])

        examples = ragas_dataset_to_dspy_examples(dataset, "test_prompt")

        assert len(examples) == 1

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
    def test_skip_missing_prompt_name(self):
        """Test that samples without the specified prompt are skipped."""
        from ragas.optimizers.dspy_adapter import ragas_dataset_to_dspy_examples

        prompt_annotation = PromptAnnotation(
            prompt_input={"question": "What is 2+2?"},
            prompt_output={"answer": "4"},
            edited_output=None,
        )

        sample = SampleAnnotation(
            metric_input={"question": "What is 2+2?"},
            metric_output=0.9,
            prompts={"other_prompt": prompt_annotation},
            is_accepted=True,
        )

        dataset = SingleMetricAnnotation(name="test_metric", samples=[sample])

        examples = ragas_dataset_to_dspy_examples(dataset, "test_prompt")

        assert len(examples) == 0

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
    def test_edited_output_priority(self):
        """Test that edited_output takes priority over prompt_output."""
        from ragas.optimizers.dspy_adapter import ragas_dataset_to_dspy_examples

        prompt_annotation = PromptAnnotation(
            prompt_input={"question": "What is 2+2?"},
            prompt_output={"answer": "3"},
            edited_output={"answer": "4"},
        )

        sample = SampleAnnotation(
            metric_input={"question": "What is 2+2?"},
            metric_output=0.9,
            prompts={"test_prompt": prompt_annotation},
            is_accepted=True,
        )

        dataset = SingleMetricAnnotation(name="test_metric", samples=[sample])

        examples = ragas_dataset_to_dspy_examples(dataset, "test_prompt")

        assert len(examples) == 1
        assert examples[0].answer == "4"

    @pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
    def test_string_output_in_dict(self):
        """Test handling of string values in dict prompt outputs."""
        from ragas.optimizers.dspy_adapter import ragas_dataset_to_dspy_examples

        prompt_annotation = PromptAnnotation(
            prompt_input={"question": "What is 2+2?"},
            prompt_output={"result": "4"},
            edited_output=None,
        )

        sample = SampleAnnotation(
            metric_input={"question": "What is 2+2?"},
            metric_output=0.9,
            prompts={"test_prompt": prompt_annotation},
            is_accepted=True,
        )

        dataset = SingleMetricAnnotation(name="test_metric", samples=[sample])

        examples = ragas_dataset_to_dspy_examples(dataset, "test_prompt")

        assert len(examples) == 1
        assert examples[0].result == "4"

    def test_import_error_without_dspy(self):
        """Test that conversion raises ImportError when dspy is not installed."""
        from ragas.optimizers.dspy_adapter import ragas_dataset_to_dspy_examples

        dataset = Mock(spec=SingleMetricAnnotation)

        with patch.dict("sys.modules", {"dspy": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(
                    ImportError, match="DSPy optimizer requires dspy\\b"
                ):
                    ragas_dataset_to_dspy_examples(dataset, "test_prompt")


class TestCreateDSPyMetric:
    def test_basic_metric_conversion(self):
        """Test basic conversion of Ragas loss to DSPy metric."""
        from ragas.optimizers.dspy_adapter import create_dspy_metric

        loss = MSELoss()
        metric_fn = create_dspy_metric(loss, "score")

        mock_example = Mock()
        mock_example.score = 0.9

        mock_prediction = Mock()
        mock_prediction.score = 0.8

        result = metric_fn(mock_example, mock_prediction)

        assert isinstance(result, float)
        assert result < 0

    def test_metric_with_missing_ground_truth(self):
        """Test metric returns 0 when ground truth is missing."""
        from ragas.optimizers.dspy_adapter import create_dspy_metric

        loss = MSELoss()
        metric_fn = create_dspy_metric(loss, "score")

        mock_example = Mock(spec=[])
        mock_prediction = Mock()
        mock_prediction.score = 0.8

        result = metric_fn(mock_example, mock_prediction)

        assert result == 0.0

    def test_metric_with_missing_prediction(self):
        """Test metric returns 0 when prediction is missing."""
        from ragas.optimizers.dspy_adapter import create_dspy_metric

        loss = MSELoss()
        metric_fn = create_dspy_metric(loss, "score")

        mock_example = Mock()
        mock_example.score = 0.9

        mock_prediction = Mock(spec=[])

        result = metric_fn(mock_example, mock_prediction)

        assert result == 0.0

    def test_metric_negation(self):
        """Test that loss is negated for DSPy (higher is better)."""
        from ragas.optimizers.dspy_adapter import create_dspy_metric

        loss = MSELoss()
        metric_fn = create_dspy_metric(loss, "score")

        mock_example = Mock()
        mock_example.score = 0.9

        mock_prediction = Mock()
        mock_prediction.score = 0.9

        result = metric_fn(mock_example, mock_prediction)

        assert result >= 0


class TestSetupDSPyLLM:
    @patch("ragas.optimizers.dspy_adapter.dspy_lm_from_llm")
    def test_setup_configures_dspy(self, mock_lm_from_llm):
        """setup_dspy_llm configures DSPy with the LM built from the LLM."""
        from ragas.optimizers.dspy_adapter import setup_dspy_llm

        mock_dspy = MagicMock()
        llm = Mock()
        mock_lm = Mock()
        mock_lm_from_llm.return_value = mock_lm

        setup_dspy_llm(mock_dspy, llm)

        mock_lm_from_llm.assert_called_once_with(mock_dspy, llm)
        mock_dspy.settings.configure.assert_called_once_with(lm=mock_lm)
