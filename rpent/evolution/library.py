"""Immutable rendered-memory snapshots and exact Markdown patches."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rpent.evolution.schemas import SkillOverlayPatch, SkillPatch

MANIFEST_NAME = "library.json"
MEMORY_DIR_NAME = "rendered_memory"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _assert_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"immutable library already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_target(memory_root: Path, target: str) -> Path:
    relative = Path(target)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"patch target must be a relative path: {target}")
    if relative.suffix.lower() != ".md":
        raise ValueError("skill patches may only target Markdown files")
    resolved = (memory_root / relative).resolve()
    try:
        resolved.relative_to(memory_root.resolve())
    except ValueError as exc:
        raise ValueError(f"patch target escapes memory root: {target}") from exc
    return resolved


def load_manifest(library_dir: str | Path) -> dict[str, Any]:
    path = Path(library_dir).resolve() / MANIFEST_NAME
    value = json.loads(path.read_text())
    if value.get("schema_version") != "SkillLibrary/v1":
        raise ValueError(f"unsupported skill library manifest: {path}")
    return value


def rendered_memory_dir(library_dir: str | Path) -> Path:
    root = Path(library_dir).resolve()
    load_manifest(root)
    memory = root / MEMORY_DIR_NAME
    if not (memory / "MEMORY.md").is_file():
        raise ValueError(f"library has no rendered MEMORY.md: {root}")
    return memory


def create_snapshot(
    memory_dir: str | Path,
    output_dir: str | Path,
    *,
    library_id: str = "S000",
) -> Path:
    """Copy the complete reviewed baseline memory into an immutable library."""

    source = Path(memory_dir).resolve()
    output = Path(output_dir).resolve()
    if not (source / "MEMORY.md").is_file():
        raise ValueError(f"source memory has no MEMORY.md: {source}")
    _assert_new_directory(output)
    memory_out = output / MEMORY_DIR_NAME
    shutil.copytree(source, memory_out)
    files = sorted(
        str(path.relative_to(memory_out))
        for path in memory_out.rglob("*")
        if path.is_file()
    )
    _write_json(
        output / MANIFEST_NAME,
        {
            "schema_version": "SkillLibrary/v1",
            "library_id": library_id,
            "parent_library": None,
            "created_at": _now(),
            "source_memory": str(source),
            "patch": None,
            "files": files,
        },
    )
    return output

def apply_patch(
    parent_dir: str | Path,
    patch: SkillPatch | dict[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    library_id: str,
) -> Path:
    """Materialize a complete candidate library with exactly one edit."""

    parent = Path(parent_dir).resolve()
    parent_manifest = load_manifest(parent)
    if isinstance(patch, (str, Path)):
        patch_value = SkillPatch.model_validate_json(Path(patch).read_text())
    elif isinstance(patch, dict):
        patch_value = SkillPatch.model_validate(patch)
    else:
        patch_value = patch

    output = Path(output_dir).resolve()
    _assert_new_directory(output)
    try:
        shutil.copytree(parent / MEMORY_DIR_NAME, output / MEMORY_DIR_NAME)
        target = _safe_target(output / MEMORY_DIR_NAME, patch_value.target)
        if patch_value.operation == "add":
            if target.exists():
                raise ValueError(f"add target already exists: {patch_value.target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(patch_value.new_text)
        else:
            if not target.is_file():
                raise ValueError(f"replace target does not exist: {patch_value.target}")
            original = target.read_text()
            occurrences = original.count(patch_value.old_text)
            if occurrences != 1:
                raise ValueError(
                    "replace old_text must occur exactly once; "
                    f"found {occurrences} in {patch_value.target}"
                )
            target.write_text(original.replace(patch_value.old_text, patch_value.new_text, 1))

        files = sorted(
            str(path.relative_to(output / MEMORY_DIR_NAME))
            for path in (output / MEMORY_DIR_NAME).rglob("*")
            if path.is_file()
        )
        _write_json(
            output / MANIFEST_NAME,
            {
                "schema_version": "SkillLibrary/v1",
                "library_id": library_id,
                "parent_library": parent_manifest["library_id"],
                "created_at": _now(),
                "source_memory": parent_manifest.get("source_memory"),
                "patch": patch_value.model_dump(mode="json"),
                "files": files,
            },
        )
    except Exception:
        # The directory is a not-yet-published candidate.  Remove only this
        # exact path after validating it was newly created above.
        shutil.rmtree(output)
        raise
    return output


def apply_overlay(
    parent_dir: str | Path,
    patch: SkillOverlayPatch | dict[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    library_id: str,
) -> Path:
    """Materialize one routing replacement or one managed leaf append."""
    parent = Path(parent_dir).resolve()
    parent_manifest = load_manifest(parent)
    if isinstance(patch, (str, Path)):
        value = SkillOverlayPatch.model_validate_json(Path(patch).read_text())
    elif isinstance(patch, dict):
        value = SkillOverlayPatch.model_validate(patch)
    else:
        value = patch
    output = Path(output_dir).resolve()
    _assert_new_directory(output)
    try:
        shutil.copytree(parent / MEMORY_DIR_NAME, output / MEMORY_DIR_NAME)
        target = _safe_target(output / MEMORY_DIR_NAME, value.target)
        if not target.is_file():
            raise ValueError(f"overlay target does not exist: {value.target}")
        original = target.read_text()
        if value.surface == "routing":
            if original.count(value.old_text) != 1:
                raise ValueError("routing overlay old_text must occur exactly once")
            rendered = original.replace(value.old_text, value.new_text, 1)
        else:
            if value.old_text:
                raise ValueError("leaf overlay is append-only and must not contain old_text")
            if value.new_text.strip() in original:
                raise ValueError("leaf overlay duplicates an existing managed record")
            rendered = original.rstrip() + value.new_text
        target.write_text(rendered)
        overlay_dir = output / "overlays"
        overlay_dir.mkdir()
        _write_json(overlay_dir / f"{value.patch_id}.json", value.model_dump(mode="json"))
        files = sorted(
            str(path.relative_to(output))
            for path in output.rglob("*")
            if path.is_file() and path.name != MANIFEST_NAME
        )
        _write_json(
            output / MANIFEST_NAME,
            {
                "schema_version": "SkillLibrary/v1",
                "library_id": library_id,
                "parent_library": parent_manifest["library_id"],
                "created_at": _now(),
                "source_memory": parent_manifest.get("source_memory"),
                "patch": value.model_dump(mode="json"),
                "files": files,
            },
        )
    except Exception:
        shutil.rmtree(output)
        raise
    return output
