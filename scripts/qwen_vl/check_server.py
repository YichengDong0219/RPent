#!/usr/bin/env python3
"""Check that a Qwen-VL server accepts images and emits OpenAI tool calls."""

from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import os
import struct
import sys
import zlib
from urllib.parse import urlsplit

import httpx


def _red_png_data_url(size: int = 64) -> str:
    """Create a dependency-free solid red RGB PNG as a data URL."""
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    row = b"\x00" + (b"\xff\x00\x00" * size)
    raw = row * size
    png = (
        signature
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _request(
    client: httpx.Client,
    base_url: str,
    model: str,
    payload: dict,
) -> dict:
    response = client.post(
        f"{base_url}/chat/completions",
        json={"model": model, **payload},
    )
    if response.is_error:
        raise RuntimeError(
            f"POST /chat/completions returned {response.status_code}: "
            f"{response.text[:1000]}"
        )
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Qwen-VL OpenAI image and tool-call compatibility."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN_VL_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible API root.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("QWEN_VL_API_KEY", "EMPTY"),
        help="Bearer key configured on the vLLM server.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_VL_MODEL"),
        help="Served model name; defaults to the first entry from /v1/models.",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.api_key}"}

    try:
        host = urlsplit(base_url).hostname or ""
        try:
            local_endpoint = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
        except ValueError:
            local_endpoint = host.lower() == "localhost"
        with httpx.Client(
            headers=headers,
            timeout=args.timeout,
            trust_env=not local_endpoint,
        ) as client:
            models_response = client.get(f"{base_url}/models")
            models_response.raise_for_status()
            models = models_response.json().get("data") or []
            model = args.model or (models[0].get("id") if models else None)
            if not model:
                raise RuntimeError("GET /models returned no model ids")
            print(f"models endpoint: OK ({model})")

            thinking = {"enable_thinking": True} if args.enable_thinking else {}
            vision = _request(
                client,
                base_url,
                model,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Identify the dominant color in this image. "
                                        "Reply with one short word."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": _red_png_data_url()},
                                },
                            ],
                        }
                    ],
                    # Qwen3.5 emits reasoning before its final answer.
                    "max_tokens": 2048 if args.enable_thinking else 512,
                    **thinking,
                },
            )
            vision_text = (
                vision.get("choices", [{}])[0].get("message", {}).get("content")
            )
            if not vision_text:
                raise RuntimeError(f"vision response had no text: {vision}")
            print(f"image input: OK ({vision_text!r})")

            tool_result = _request(
                client,
                base_url,
                model,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Call report_status exactly once with status "
                                "set to ready."
                            ),
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "report_status",
                                "description": "Report the service status.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                    },
                                    "required": ["status"],
                                    "additionalProperties": False,
                                },
                            },
                        }
                    ],
                    "tool_choice": "auto",
                    # Leave enough room for reasoning plus the structured call.
                    "max_tokens": 2048 if args.enable_thinking else 512,
                    **thinking,
                },
            )
            message = tool_result.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                raise RuntimeError(
                    "model returned no parsed tool_calls; check "
                    "--enable-auto-tool-choice and --tool-call-parser. "
                    f"Response: {json.dumps(tool_result, ensure_ascii=False)[:2000]}"
                )
            call = tool_calls[0]
            function = call.get("function") or {}
            arguments = json.loads(function.get("arguments") or "{}")
            if function.get("name") != "report_status":
                raise RuntimeError(f"unexpected tool name: {function!r}")
            print(f"tool calling: OK ({arguments})")
    except (httpx.HTTPError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Qwen-VL server check failed: {exc}", file=sys.stderr)
        return 1

    print("Qwen-VL service is compatible with the RPent API planner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
