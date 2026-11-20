"""Baseline QA pipelines (P0, P0-raw, P1 ReAct) and shared prompts."""

from src.baselines.p0 import run_p0, run_p0_raw
from src.baselines.p1_react import run_p1
from src.baselines.prompts import (
    CONTENT_INSTRUCTION,
    FORMAT_LINE,
    P1_TOOL_GUIDANCE,
    SHARED_ANSWER_INSTRUCTION,
    build_p0_prompt,
    build_p0_raw_prompt,
    build_p1_system_prompt,
)

__all__ = [
    "CONTENT_INSTRUCTION",
    "FORMAT_LINE",
    "P1_TOOL_GUIDANCE",
    "SHARED_ANSWER_INSTRUCTION",
    "build_p0_prompt",
    "build_p0_raw_prompt",
    "build_p1_system_prompt",
    "run_p0",
    "run_p0_raw",
    "run_p1",
]
