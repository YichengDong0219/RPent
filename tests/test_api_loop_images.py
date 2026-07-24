from __future__ import annotations

from pathlib import Path

from pydantic_ai import ToolReturn

from rpent.planner.api_loop import read_image


def _write_png_stub(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


def test_read_image_uses_existing_path(tmp_path: Path) -> None:
    image_path = tmp_path / "images_wrist" / "image_wrist_00.png"
    _write_png_stub(image_path)

    result = read_image(str(image_path))

    assert isinstance(result, ToolReturn)
    assert result.return_value == str(image_path)


def test_read_image_corrects_split_output_directory(tmp_path: Path) -> None:
    image_path = tmp_path / "images_wrist" / "image_wrist_00.png"
    _write_png_stub(image_path)
    misspelled = tmp_path / "images" / "wrist" / image_path.name

    result = read_image(str(misspelled))

    assert isinstance(result, ToolReturn)
    assert result.return_value == str(image_path)


def test_read_image_missing_file_is_recoverable(tmp_path: Path) -> None:
    available = tmp_path / "images_cam" / "frame.png"
    _write_png_stub(available)
    missing = tmp_path / "images" / "wrist" / "frame.png"

    result = read_image(str(missing))

    assert isinstance(result, str)
    assert "Image file not found" in result
    assert str(available) in result
    assert "call read_image again" in result
