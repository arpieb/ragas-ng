"""Golden tests pinning the shape of the ragas run tree.

Stage 5 of the LangChain removal replaces the run-tree primitive in
``ragas.callbacks`` -- previously built on ``langchain_core.callbacks`` -- with a
native implementation. The risk is not that it breaks loudly; it is that the
trace tree drifts silently and the parsed traces change shape underneath
``optimizers/genetic.py`` and anyone else reading them.

These tests were written and run against the LangChain implementation first, so
they describe the behaviour that must be preserved rather than the behaviour that
happens to exist afterwards.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import pytest

from ragas.callbacks import ChainRun, ChainType, RagasTracer, new_group


def tree_shape(traces: t.Dict[str, ChainRun]) -> t.List[t.Dict[str, t.Any]]:
    """Render a trace tree with volatile ids replaced by stable indices.

    Run ids are uuids, so the raw dict is never comparable between runs. This
    keeps names, nesting, metadata types and outputs -- everything that is
    actually part of the contract.
    """
    order = {run_id: i for i, run_id in enumerate(traces)}

    def render(run: ChainRun) -> t.Dict[str, t.Any]:
        meta = dict(run.metadata)
        if isinstance(meta.get("type"), ChainType):
            meta["type"] = meta["type"].value
        return {
            "idx": order[run.run_id],
            "parent": order[run.parent_run_id] if run.parent_run_id else None,
            "name": run.name,
            "inputs": run.inputs,
            "metadata": meta,
            "outputs": run.outputs,
            "children": sorted(order[c] for c in run.children),
        }

    return [render(r) for r in traces.values()]


class TestNewGroupNesting:
    """The primitive itself: parent/child wiring, outputs, and the ended flag."""

    def test_single_group_records_start_and_end(self):
        tracer = RagasTracer()
        rm, group = new_group(
            name="root",
            inputs={"a": 1},
            callbacks=[tracer],
            metadata={"type": ChainType.EVALUATION},
        )
        assert group.ended is False
        rm.on_chain_end({"scores": [1.0]})

        assert tree_shape(tracer.traces) == [
            {
                "idx": 0,
                "parent": None,
                "name": "root",
                "inputs": {"a": 1},
                "metadata": {"type": "evaluation"},
                "outputs": {"scores": [1.0]},
                "children": [],
            }
        ]

    def test_nested_groups_link_parent_to_child(self):
        """A child group is created by passing the parent group as callbacks."""
        tracer = RagasTracer()
        root_rm, root_group = new_group(
            name="root", inputs={}, callbacks=[tracer], metadata={}
        )
        child_rm, child_group = new_group(
            name="row 0", inputs={"q": "?"}, callbacks=root_group, metadata={}
        )
        grandchild_rm, _ = new_group(
            name="metric", inputs={}, callbacks=child_group, metadata={}
        )

        grandchild_rm.on_chain_end({"output": 0.5})
        child_rm.on_chain_end({"done": True})
        root_rm.on_chain_end({})

        shape = tree_shape(tracer.traces)
        assert [s["name"] for s in shape] == ["root", "row 0", "metric"]
        assert [s["parent"] for s in shape] == [None, 0, 1]
        assert shape[0]["children"] == [1]
        assert shape[1]["children"] == [2]
        assert shape[2]["outputs"] == {"output": 0.5}

    def test_outputs_accepted_positionally_and_by_keyword(self):
        """Both call styles are used across the 21 existing call sites."""
        tracer = RagasTracer()
        rm_a, _ = new_group(name="a", inputs={}, callbacks=[tracer], metadata={})
        rm_a.on_chain_end({"positional": True})
        rm_b, _ = new_group(name="b", inputs={}, callbacks=[tracer], metadata={})
        rm_b.on_chain_end(outputs={"keyword": True})

        outs = [r.outputs for r in tracer.traces.values()]
        assert {"positional": True} in outs
        assert {"keyword": True} in outs

    def test_run_manager_exposes_run_id(self):
        """optimizers/genetic.py reads rm.run_id directly."""
        tracer = RagasTracer()
        rm, _ = new_group(name="x", inputs={}, callbacks=[tracer], metadata={})
        assert str(rm.run_id) in tracer.traces

    def test_outputs_preserve_live_python_objects(self):
        """The reason the native tree exists at all.

        Chain outputs carry Testset objects, pydantic models and score dicts.
        They must come back out as the same objects, not serialised copies --
        optimizers/genetic.py calls .model_dump() on what it finds here.
        """

        class Rich:
            pass

        obj = Rich()
        tracer = RagasTracer()
        rm, _ = new_group(name="x", inputs={}, callbacks=[tracer], metadata={})
        rm.on_chain_end({"output": obj})

        assert tracer.traces[str(rm.run_id)].outputs["output"] is obj


class TestErrorHandling:
    def test_on_chain_error_is_accepted_positionally_and_by_keyword(self):
        tracer = RagasTracer()
        rm_a, _ = new_group(name="a", inputs={}, callbacks=[tracer], metadata={})
        rm_a.on_chain_error(ValueError("boom"))
        rm_b, _ = new_group(name="b", inputs={}, callbacks=[tracer], metadata={})
        rm_b.on_chain_error(error=ValueError("boom"))

    def test_a_raising_handler_does_not_abort_the_run(self):
        """A broken tracer must never take down an evaluation."""

        @dataclass
        class Exploding:
            seen: t.List[str] = field(default_factory=list)

            def on_chain_start(self, serialized, inputs, **kw):
                raise RuntimeError("handler is broken")

            def on_chain_end(self, outputs, **kw):
                raise RuntimeError("handler is broken")

        tracer = RagasTracer()
        rm, group = new_group(
            name="root", inputs={}, callbacks=[Exploding(), tracer], metadata={}
        )
        rm.on_chain_end({"ok": True})

        # the healthy tracer still saw everything
        assert [r.name for r in tracer.traces.values()] == ["root"]
        assert list(tracer.traces.values())[0].outputs == {"ok": True}


class TestParseRunTraces:
    def test_single_root_invariant(self):
        """parse_run_traces refuses a forest; two roots is a bug signal."""
        from ragas.callbacks import parse_run_traces

        tracer = RagasTracer()
        new_group(name="root-a", inputs={}, callbacks=[tracer], metadata={})
        new_group(name="root-b", inputs={}, callbacks=[tracer], metadata={})

        with pytest.raises(ValueError, match="Multiple root traces"):
            parse_run_traces(tracer.traces)

    def test_evaluation_row_metric_prompt_nesting(self):
        """The four-level shape parse_run_traces expects, built explicitly."""
        from ragas.callbacks import parse_run_traces

        tracer = RagasTracer()
        root_rm, root_group = new_group(
            name="ragas evaluation",
            inputs={},
            callbacks=[tracer],
            metadata={"type": ChainType.EVALUATION},
        )
        row_rm, row_group = new_group(
            name="row 0",
            inputs={"user_input": "q"},
            callbacks=root_group,
            metadata={"type": ChainType.ROW, "row_index": 0},
        )
        metric_rm, metric_group = new_group(
            name="faithfulness",
            inputs={},
            callbacks=row_group,
            metadata={"type": ChainType.METRIC},
        )
        prompt_rm, _ = new_group(
            name="StatementGenerator",
            inputs={"data": {"question": "q"}},
            callbacks=metric_group,
            metadata={"type": ChainType.RAGAS_PROMPT},
        )
        prompt_rm.on_chain_end({"output": ["generated"]})
        metric_rm.on_chain_end({"output": 0.75})
        row_rm.on_chain_end({"faithfulness": 0.75})
        root_rm.on_chain_end({"scores": []})

        parsed = parse_run_traces(tracer.traces)
        assert len(parsed) == 1
        assert parsed[0].scores == {"faithfulness": 0.75}
        assert parsed[0]["faithfulness"] == {
            "StatementGenerator": {
                "input": {"question": "q"},
                "output": "generated",
            }
        }


def test_no_traces_parses_to_nothing():
    """An empty trace tree is not an error.

    ``parse_run_traces`` guarded against more than one root but not against
    none, so parsing a tree with nothing under the parent (e.g. the old
    ``EvaluationResult`` built with only its required fields) raised IndexError.
    """
    from ragas.callbacks import parse_run_traces

    assert parse_run_traces({}) == []
