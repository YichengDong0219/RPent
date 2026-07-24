"""LIBERO prompt bundle assembly."""
from __future__ import annotations

from pathlib import Path

from robots.libero import prompts as libero_prompt
from rpent.context.prompt_utils import PromptNode
from rpent.context.prompts import prompt as base_prompt
from rpent.utils.resources import RESOURCES_ABLATION_ENV, resource_context_disabled

_RESOURCE_ABLATION_NOTICE = f"""
═══════════════════════════════════════════════════════════════════════
RESOURCE-ABLATION MODE — OVERRIDES RESOURCE-READING RULES BELOW
═══════════════════════════════════════════════════════════════════════
This run intentionally excludes RPent memory, seed recipes, and guide-file
context ({RESOURCES_ABLATION_ENV}=1). Do not call read_text_file or list_dir on
`resources/libero/**` or `robots/libero/guides/**`, even where later sections
say those reads are mandatory. The operational rules needed for this smoke run
are already embedded in this system prompt.

Begin immediately with `view_driver_state({{"step": 0}})`, read the exact
high-resolution image path it returns, localize by perception/back-projection,
and execute the task. Do not invent or rewrite returned file paths. In the final
strategy notes, record that this was a resource-ablation run.

"""


def system_prompt() -> PromptNode:
    """Return the system prompt text."""
    prompt = (
        Path(__file__).parent / "prompts" / "perception_system_prompt.md"
    ).read_text(encoding="utf-8")
    if resource_context_disabled():
        return _RESOURCE_ABLATION_NOTICE + prompt
    return prompt

    # Previous sectioned prompt kept for reference while the aligned perception
    # prompt is reviewed:
    # return {
    #     "Intro": libero_prompt.PREAMBLE,
    #     "Goal": libero_prompt.GOAL,
    #     "Rules": libero_prompt.RULES,
    #     "Localization": libero_prompt.LOCALIZATION,
    #     "Workflow": libero_prompt.WORKFLOW,
    #     "Environment": libero_prompt.ENVIRONMENT,
    #     "Output": base_prompt.OUTPUT,
    #     "Next": libero_prompt.NEXT,
    # }


def user_prompt() -> dict[str, PromptNode]:
    """Return the first user message tree."""
    sections = dict(base_prompt.USER)
    sections["Mode"] = libero_prompt.USER_MODE
    return sections
