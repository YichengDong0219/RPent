Agentic Planner
===============

RPent's reasoning brain — the *planner* — is chosen with a single CLI
flag:

.. code-block:: bash

   --planner {api, claude_code, codex}

All three planners see the same tool schemas and the same prompt
bundle. They differ only in *how* the tool-calling loop is orchestrated
and in *which* LLMs / SDKs they can reach.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - ``--planner``
     - What it is
     - When to pick it
   * - ``api``
     - Provider-agnostic tool-calling loop built on
       `pydantic-ai <https://ai.pydantic.dev/>`_. Talks Anthropic,
       OpenAI Responses, and OpenAI-compatible chat endpoints.
       Handles prompt caching and history-image pruning.
     - You want the tightest control over model calls, the widest
       provider coverage, or the cheapest per-turn spend.
   * - ``claude_code``
     - The `Claude Agent SDK
       <https://docs.claude.com/en/api/agent-sdk/overview>`_. Exposes
       RPent's toolkit as an in-process MCP server; Claude drives the
       loop.
     - You want Claude's native agent runtime (memory,
       thinking-mode budgets, robust tool retries).
   * - ``codex``
     - The OpenAI **Codex SDK**, bridged to the toolkit over an HTTP
       MCP server.
     - You want the Codex agent runtime or you already have OpenAI /
       Codex quota to spend.

The ``api`` planner (custom / lightweight)
-------------------------------------------

``--planner api`` runs a hand-rolled pydantic-ai loop. It is the
default and the most portable — any provider that speaks the Anthropic
Messages API, the OpenAI Responses API, or an OpenAI-compatible chat
API works.

Pick the provider by prefixing ``--model``:

.. code-block:: bash

   # Anthropic Claude
   rpent --planner api --model anthropic:claude-opus-4-8 ...

   # OpenAI Responses (e.g. GPT-5.5)
   rpent --planner api --model openai:gpt-5.5 ...

   # OpenAI-compatible chat (e.g. GLM 5.2, text-only)
   rpent --planner api --model openai-chat:glm-5.2 --no-images ...

   # Local Qwen3.5 (OpenAI-compatible vLLM service, with image input)
   rpent --planner api \
     --model qwen-vl:Qwen3.5-9B \
     --base-url http://127.0.0.1:8000/v1 ...

Environment variables it reads (override with ``--base-url`` if
needed):

- ``anthropic:*`` → ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY``
- ``openai:*`` / ``openai-chat:*`` → ``OPENAI_BASE_URL`` /
  ``OPENAI_API_KEY``
- ``qwen-vl:*`` → ``QWEN_VL_BASE_URL`` / ``QWEN_VL_API_KEY``.
  The key defaults to ``EMPTY`` for an unauthenticated local server.

Useful ``api``-only knobs:

- ``--max-tokens`` — cap each LLM reply (default ``8192``).
- ``--max-turns`` — cap the number of tool-calling turns (default
  ``100``).
- ``--no-images`` — never send image bytes; add it for text-only
  models, which otherwise fail with
  ``400 "message type 'image_url' is not supported"``. The agent then
  reasons from textual state alone, so task performance may not be
  satisfactory.

The ``claude_code`` planner
----------------------------

``--planner claude_code`` delegates the loop to the Claude Agent SDK.
RPent's tools become an **in-process MCP server** that Claude Code
calls; you see the same tools under the ``mcp__rpent__<name>``
namespace.

.. code-block:: bash

   rpent --planner claude_code \
     --model claude-opus-4-8 \
     --suite libero_object_swap --task 2 --seed 0

Notes:

- Do **not** prefix the model id with a provider — pass e.g.
  ``claude-opus-4-8``.
- The subprocess is capped by a wall-clock budget
  (``--planner-timeout-s``, defaults to
  ``CODEX_TIMEOUT_S`` / ``CELL_TIMEOUT_S`` / ``1200``).
- A dollar budget can be set via ``--claude-code-max-budget-usd``
  (defaults to ``MAX_BUDGET_USD`` env or ``10``).
- Claude Code needs to be installed and authenticated separately; see
  the `Claude Agent SDK docs
  <https://docs.claude.com/en/api/agent-sdk/overview>`_.

The ``codex`` planner
----------------------

``--planner codex`` bridges the same toolkit to the OpenAI Codex SDK
over an HTTP MCP server started by ``scripts/codex_proxy/``.

.. code-block:: bash

   rpent --planner codex \
     --model gpt-5.5 \
     --suite libero_goal_task --task 1 --seed 0

Notes:

- ``--planner-timeout-s`` bounds the Codex subprocess in the same way
  as ``claude_code``.
- Codex authentication uses the standard OpenAI environment
  variables.

Bring your own agent
--------------------

If none of the three planners fit — say you want to plug in an
in-house planner, a research prototype, or a different agent SDK —
subclass ``rpent.planner.base.Planner`` and register your factory in
``rpent.planner.base.build_planner``:

.. code-block:: python

   # rpent/planner/mybrain.py
   from rpent.planner.base import Planner

   class MyPlanner(Planner):
       async def run(self, *, prompt_bundle, toolkit, output_dir, ...):
           # Drive the tool-calling loop yourself.
           # Call toolkit.dispatch(tool_name, **kwargs) to invoke a tool.
           ...

Any planner must:

1. Take the rendered ``prompt_bundle`` (system + user prompt sections
   from ``robots/<env>/prompt_bundle.py``).
2. Loop over LLM replies, extract tool calls, and forward them to the
   toolkit via ``toolkit.dispatch(...)``.
3. Feed each tool's return value back into the LLM as multimodal
   context (text + images).
4. Terminate on ``finish`` or when the caps are hit.

Because every planner sees the same schemas and the same prompts,
adding a new brain never requires touching the tools or the env
servers. See :doc:`../development/architecture` for the interface, and
:doc:`../development/add_primitive` if you want to expose new tools to
your custom brain.

Choosing max-tokens and max-turns
---------------------------------

Two knobs bound every planner run:

- ``--max-tokens`` caps *per-reply* tokens. LIBERO-style tasks usually
  finish comfortably under ``8192``; longer-horizon RoboCasa episodes
  benefit from raising it if your model supports it.
- ``--max-turns`` caps the *total number of tool-calling turns*. A
  single LIBERO task rarely needs more than ~30 turns; RoboCasa
  long-horizon tasks can approach the default ``100``.

Both caps trigger a graceful ``finish(stuck)`` outcome rather than a
hard crash, so you can tune them without losing the transcript.

Local Qwen3.5 service
---------------------

The ``qwen-vl:`` provider connects to a local multimodal service through
OpenAI Chat Completions. RPent camera frames are sent as base64
``image_url`` blocks while the existing ``api`` planner continues to manage
tool schemas and tool results. Do not pass ``--no-images`` with Qwen-VL.

The client defaults match the existing 9B launcher:

.. code-block:: bash

   # Qwen/Qwen3.5-9B, served as Qwen3.5-9B on GPU 0 and port 8000
   bash /home/dongyicheng/set_qwen9b.sh

The smaller existing launcher uses a different served name and endpoint:

.. code-block:: bash

   # Qwen/Qwen3.5-4B, served as Qwen3.5-4B on GPU 3 and port 8001
   bash /home/dongyicheng/set_qwen4b.sh
   export QWEN_VL_MODEL=Qwen3.5-4B
   export QWEN_VL_BASE_URL=http://127.0.0.1:8001/v1

Before booting the simulator, validate model discovery, base64 image input,
and parsed OpenAI tool calls:

.. code-block:: bash

   conda activate rpent
   cd /home/dongyicheng/rpent
   python scripts/qwen_vl/check_server.py \
     --base-url http://127.0.0.1:8000/v1

The existing launchers enable ``--reasoning-parser qwen3``,
``--enable-auto-tool-choice``, and ``--tool-call-parser qwen3_coder``.
Those flags are required: an ordinary image-chat endpoint works without them,
but RPent cannot execute simulator tools unless vLLM returns structured
``tool_calls``.

Then launch RPent:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   export CUDA_VISIBLE_DEVICES=<pi05-gpu>
   export QWEN_VL_BASE_URL=http://127.0.0.1:8000/v1
   export QWEN_VL_API_KEY=EMPTY

   rpent --env libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner api \
     --model qwen-vl:Qwen3.5-9B \
     --max-tokens 4096

The 9B launcher exposes a 32K context window. The initial RPent prompt and
tool schemas use about 8K tokens, so a per-reply cap of 4K leaves room for
multi-turn tool results. Do not use ``--max-tokens 8192`` with the 16K 4B
launcher: input plus the requested output budget can exceed its context limit.

Qwen3.5 and Pi0.5 must fit in GPU memory at the same time. The current 9B
launcher uses physical GPU 0; the 4B launcher uses physical GPU 3. Choose a
different free GPU for Pi0.5 when possible. Both launchers bind vLLM to
``0.0.0.0`` without an API key, so restrict ports 8000/8001 with a firewall
if the host is reachable from an untrusted network.
