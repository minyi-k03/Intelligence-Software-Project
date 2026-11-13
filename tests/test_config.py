"""Unit tests for configuration, callbacks, and LLM factory.

References:
- AGENTS.md
- docs/INTERFACES.md §2, §4
"""

from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.common.types import CallRecord, PipelineResult
from src.config.callbacks import MetricsCallbackHandler
from src.config.llm import create_chat_ollama, get_llm_kwargs
from src.config.settings import (
    ModelConfig,
    Settings,
    get_settings,
    load_settings,
)


class TestConfig(unittest.TestCase):
    def test_default_yaml_loading(self):
        """Verify that configs/default.yaml loads properly with exact interface values."""
        settings = load_settings("configs/default.yaml")

        self.assertEqual(settings.model.tag, "qwen2.5:7b-instruct-q4_K_M")
        self.assertEqual(settings.model.temperature, 0)
        self.assertEqual(settings.model.num_ctx, 8192)
        self.assertEqual(settings.model.keep_alive, -1)

        self.assertEqual(settings.budget.max_calls, 4)
        self.assertEqual(settings.budget.max_requery, 2)

        self.assertEqual(settings.cap.first, 512)
        self.assertEqual(settings.cap.finalize, 128)
        self.assertEqual(settings.cap.judge, 256)
        self.assertEqual(settings.cap.retry_multiplier, 2)

        self.assertEqual(settings.retrieval.k, 3)
        self.assertEqual(settings.compress.m, 3)
        self.assertEqual(settings.compress.window, 1)
        self.assertEqual(settings.p1.max_steps, 7)
        self.assertEqual(settings.run.timeout_s, 120)
        self.assertEqual(settings.data.seed, 42)

    def test_no_hardcoded_dataclass_defaults(self):
        """Config dataclasses must require explicit values (no hardcoded defaults in code)."""
        with self.assertRaises(TypeError):
            ModelConfig()  # type: ignore
        with self.assertRaises(TypeError):
            Settings()  # type: ignore

    def test_temperature_must_be_zero_rule(self):
        """AGENTS.md absolute rule: temperature must be 0 (and not bool False)."""
        with self.assertRaises(ValueError):
            ModelConfig(tag="dummy-tag", temperature=1, num_ctx=8192, keep_alive=-1)
        with self.assertRaises(ValueError):
            ModelConfig(tag="dummy-tag", temperature=False, num_ctx=8192, keep_alive=-1)  # type: ignore

    def test_missing_empty_or_partial_config_raises(self):
        """Empty YAML, missing sections, missing keys, or unknown sections must raise immediately."""
        with self.assertRaises(FileNotFoundError):
            load_settings("configs/nonexistent_config_file.yaml")

        # Empty YAML
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            tf.write("")
            empty_path = Path(tf.name)
        try:
            with self.assertRaises(ValueError):
                load_settings(empty_path)
        finally:
            empty_path.unlink(missing_ok=True)

        # Missing section (only model provided)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            tf.write("model:\n  tag: dummy\n  temperature: 0\n  num_ctx: 8192\n  keep_alive: -1\n")
            partial_section_path = Path(tf.name)
        try:
            with self.assertRaises(ValueError):
                load_settings(partial_section_path)
        finally:
            partial_section_path.unlink(missing_ok=True)

        # Missing key inside a section
        full_text = Path("configs/default.yaml").read_text(encoding="utf-8")
        missing_key_text = "\n".join(
            line for line in full_text.splitlines() if "keep_alive:" not in line
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            tf.write(missing_key_text)
            missing_key_path = Path(tf.name)
        try:
            with self.assertRaises(ValueError):
                load_settings(missing_key_path)
        finally:
            missing_key_path.unlink(missing_ok=True)

    def test_settings_dot_key_and_dict_snapshot(self):
        """Test dotpath property lookup and RUN_META snapshot serialization."""
        settings = get_settings(reload=True)
        self.assertEqual(settings.get("model.num_ctx"), 8192)
        self.assertEqual(settings.get("cap.first"), 512)
        self.assertEqual(settings.get("nonexistent.key", "default_val"), "default_val")

        snap = settings.to_dict()
        self.assertEqual(snap["model"]["temperature"], 0)
        self.assertEqual(snap["model"]["keep_alive"], -1)
        self.assertEqual(snap["budget"]["max_calls"], 4)

    def test_get_llm_kwargs_and_factory_binding(self):
        """Verify that LLM kwargs and factory strictly derive from settings without fallback."""
        settings = load_settings("configs/default.yaml")
        cb = MetricsCallbackHandler()

        kwargs = get_llm_kwargs(
            settings=settings,
            num_predict=256,
            format_schema={"type": "object"},
            callbacks=[cb],
        )

        self.assertEqual(kwargs["model"], "qwen2.5:7b-instruct-q4_K_M")
        self.assertEqual(kwargs["temperature"], 0)
        self.assertEqual(kwargs["num_ctx"], 8192)
        self.assertEqual(kwargs["keep_alive"], -1)
        self.assertEqual(kwargs["num_predict"], 256)
        self.assertEqual(kwargs["format"], {"type": "object"})
        self.assertEqual(kwargs["callbacks"], [cb])

        with self.assertRaises(ValueError):
            get_llm_kwargs(settings=settings, num_predict=0)

        with patch("src.config.llm.ChatOllama", None):
            with self.assertRaises(RuntimeError):
                create_chat_ollama(settings, num_predict=512)

        with patch("src.config.llm.ChatOllama") as mock_cls:
            mock_cls.return_value = "mock_llm_instance"
            inst = create_chat_ollama(settings, num_predict=512)
            self.assertEqual(inst, "mock_llm_instance")
            mock_cls.assert_called_once()
            called_kwargs = mock_cls.call_args.kwargs
            self.assertEqual(called_kwargs["model"], "qwen2.5:7b-instruct-q4_K_M")
            self.assertEqual(called_kwargs["temperature"], 0)
            self.assertEqual(called_kwargs["num_predict"], 512)

    def test_metrics_callback_handler_legacy_and_modern_ollama(self):
        """Verify token accounting for both llm_output and langchain-ollama AIMessage metadata."""
        handler = MetricsCallbackHandler(current_step="judge", cap=256, is_retry=False)

        handler.on_chat_model_start({}, [[]], run_id="run-1")
        mock_ai_msg = SimpleNamespace(
            content='{"reasoning": "Sufficient.", "decision": "ANSWER", "missing": "", "answer": "Paris"}',
            usage_metadata={"input_tokens": 145, "output_tokens": 28, "total_tokens": 173},
            response_metadata={"done_reason": "stop", "prompt_eval_count": 145, "eval_count": 28},
        )
        mock_gen = SimpleNamespace(message=mock_ai_msg, text="", generation_info=None)
        mock_resp = SimpleNamespace(generations=[[mock_gen]], llm_output=None)
        handler.on_llm_end(mock_resp, run_id="run-1")
        handler.record_parse_result(True)

        self.assertEqual(handler.n_calls, 1)
        self.assertEqual(handler.n_budget_calls, 1)
        self.assertEqual(handler.n_retries, 0)
        self.assertEqual(handler.total_input_tokens, 145)
        self.assertEqual(handler.total_output_tokens, 28)
        self.assertEqual(handler.max_input_tokens, 145)

        rec = handler.calls[0]
        self.assertEqual(rec.step, "judge")
        self.assertEqual(rec.cap, 256)
        self.assertEqual(rec.input_tokens, 145)
        self.assertEqual(rec.output_tokens, 28)
        self.assertFalse(rec.truncated)
        self.assertFalse(rec.is_retry)
        self.assertTrue(rec.parse_ok)
        self.assertGreaterEqual(rec.latency_s, 0.0)

    def test_metrics_callback_truncation_detection(self):
        """Verify that truncation is flagged when output tokens hit the cap or done_reason is length."""
        handler = MetricsCallbackHandler(current_step="router", cap=512, is_retry=False)
        handler.on_llm_start({}, ["Prompt"], run_id="run-trunc")

        mock_generation = SimpleNamespace(
            text="A" * 512,
            generation_info={"prompt_eval_count": 50, "eval_count": 512, "done_reason": "length"},
        )
        mock_response = SimpleNamespace(
            generations=[[mock_generation]],
            llm_output={"prompt_eval_count": 50, "eval_count": 512},
        )
        handler.on_llm_end(mock_response, run_id="run-trunc")
        handler.record_parse_result(False)

        self.assertTrue(handler.calls[0].truncated)
        self.assertFalse(handler.calls[0].parse_ok)

    def test_metrics_callback_auto_extracts_invoked_cap(self):
        """Verify that num_predict from invocation_params overrides/populates CallRecord.cap."""
        handler = MetricsCallbackHandler(current_step="router")
        handler.on_chat_model_start(
            {},
            [[]],
            run_id="run-inv-cap",
            invocation_params={"options": {"num_predict": 512}},
        )
        mock_ai_msg = SimpleNamespace(
            content="math route",
            usage_metadata={"input_tokens": 60, "output_tokens": 512},
            response_metadata={"done_reason": "length"},
        )
        mock_resp = SimpleNamespace(
            generations=[[SimpleNamespace(message=mock_ai_msg, text="", generation_info=None)]],
            llm_output=None,
        )
        handler.on_llm_end(mock_resp, run_id="run-inv-cap")

        self.assertEqual(handler.calls[0].cap, 512)
        self.assertTrue(handler.calls[0].truncated)

    def test_metrics_callback_retry_accounting_and_pipeline_result(self):
        """Verify that retries are excluded from n_budget_calls and PipelineResult serializes cleanly."""
        handler = MetricsCallbackHandler()

        # Call 1: Normal call (truncated)
        handler.set_call_context(step="judge", cap=256, is_retry=False)
        handler.on_llm_start({}, ["p1"])
        handler.on_llm_end(
            SimpleNamespace(
                generations=[[SimpleNamespace(text="partial", generation_info={"done_reason": "length"})]],
                llm_output={"prompt_eval_count": 100, "eval_count": 256},
            )
        )
        handler.record_parse_result(False)

        # Call 2: Retry call (cap doubled, is_retry=True)
        handler.set_call_context(step="judge", cap=512, is_retry=True)
        handler.on_llm_start({}, ["p2"])
        handler.on_llm_end(
            SimpleNamespace(
                generations=[[SimpleNamespace(text="ok", generation_info={"done_reason": "stop"})]],
                llm_output={"prompt_eval_count": 100, "eval_count": 40},
            )
        )
        handler.record_parse_result(True)

        self.assertEqual(handler.n_calls, 2)
        self.assertEqual(handler.n_budget_calls, 1)  # Retries excluded from budget
        self.assertEqual(handler.n_retries, 1)

        result = PipelineResult(
            answer="42",
            abstained=False,
            budget_exhausted=False,
            route="search",
            decisions=["ANSWER"],
            n_calls=handler.n_budget_calls,
            n_retries=handler.n_retries,
            calls=list(handler.calls),
        )
        serialized = asdict(result)
        self.assertEqual(serialized["n_calls"], 1)
        self.assertEqual(serialized["n_retries"], 1)
        self.assertEqual(len(serialized["calls"]), 2)


if __name__ == "__main__":
    unittest.main()
