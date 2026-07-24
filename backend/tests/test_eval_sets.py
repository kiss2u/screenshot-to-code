import hashlib
import json
import os
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

import evals.config
from evals.sets import (
    EvalSetNotFoundError,
    InvalidSetNameError,
    get_set,
    get_set_inputs_dir,
    list_set_images,
    list_sets,
    resolve_set_image_path,
)
from routes.eval_sets import get_eval_set_image


@pytest.fixture
def evals_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(evals.config, "EVALS_DIR", str(tmp_path))
    return tmp_path


def _make_set(evals_dir: Path, name: str, images: dict[str, bytes]) -> Path:
    inputs = evals_dir / "sets" / name / "inputs"
    inputs.mkdir(parents=True)
    for filename, content in images.items():
        (inputs / filename).write_bytes(content)
    return inputs


def test_list_set_images_bootstraps_manifest_with_hashes(evals_dir: Path) -> None:
    _make_set(evals_dir, "jun-21-evals", {"a.png": b"aaa", "b.png": b"bbb"})

    images = list_set_images("jun-21-evals")
    assert [i.filename for i in images] == ["a.png", "b.png"]
    assert images[0].sha256 == hashlib.sha256(b"aaa").hexdigest()

    manifest_path = evals_dir / "sets" / "jun-21-evals" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["display_name"] == "jun-21-evals"
    assert manifest["images"]["b.png"]["sha256"] == (
        hashlib.sha256(b"bbb").hexdigest()
    )


def test_hash_cache_reused_and_invalidated(evals_dir: Path) -> None:
    inputs = _make_set(evals_dir, "s1", {"a.png": b"original"})
    list_set_images("s1")

    # Tamper with the cached sha; with unchanged size+mtime it must be
    # served from cache (i.e. NOT recomputed).
    manifest_path = evals_dir / "sets" / "s1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["images"]["a.png"]["sha256"] = "cached-sentinel"
    manifest_path.write_text(json.dumps(manifest))
    assert list_set_images("s1")[0].sha256 == "cached-sentinel"

    # Rewriting the file (new mtime) invalidates the cache entry.
    image_path = inputs / "a.png"
    image_path.write_bytes(b"rewritten")
    os.utime(image_path, (time.time() + 5, time.time() + 5))
    images = list_set_images("s1")
    assert images[0].sha256 == hashlib.sha256(b"rewritten").hexdigest()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["images"]["a.png"]["sha256"] == images[0].sha256


def test_deleted_files_pruned_from_manifest(evals_dir: Path) -> None:
    inputs = _make_set(evals_dir, "s1", {"a.png": b"a", "b.png": b"b"})
    list_set_images("s1")
    (inputs / "b.png").unlink()

    images = list_set_images("s1")
    assert [i.filename for i in images] == ["a.png"]
    manifest = json.loads(
        (evals_dir / "sets" / "s1" / "manifest.json").read_text()
    )
    assert "b.png" not in manifest["images"]


def test_list_sets_and_get_set(evals_dir: Path) -> None:
    _make_set(evals_dir, "alpha", {"a.png": b"a"})
    _make_set(evals_dir, "beta", {"b.png": b"b", "c.png": b"c"})

    sets = list_sets()
    assert [(s.name, s.image_count) for s in sets] == [("alpha", 1), ("beta", 2)]
    assert get_set("beta").image_count == 2
    with pytest.raises(EvalSetNotFoundError):
        get_set("missing")


def test_set_name_validation(evals_dir: Path) -> None:
    for bad_name in ("../etc", "/abs", "", ".hidden"):
        with pytest.raises(InvalidSetNameError):
            get_set_inputs_dir(bad_name)


@pytest.mark.asyncio
async def test_image_path_resolution_and_route_guards(evals_dir: Path) -> None:
    _make_set(evals_dir, "s1", {"a.png": b"png-bytes"})

    assert resolve_set_image_path("s1", "a.png").endswith("a.png")
    # Traversal components are stripped to a basename.
    assert resolve_set_image_path("s1", "../a.png").endswith("a.png")
    with pytest.raises(InvalidSetNameError):
        resolve_set_image_path("s1", "notes.txt")
    with pytest.raises(FileNotFoundError):
        resolve_set_image_path("s1", "missing.png")

    response = await get_eval_set_image("s1", "a.png")
    assert response.path == str(evals_dir / "sets" / "s1" / "inputs" / "a.png")
    with pytest.raises(HTTPException) as excinfo:
        await get_eval_set_image("s1", "missing.png")
    assert excinfo.value.status_code == 404
    with pytest.raises(HTTPException) as excinfo:
        await get_eval_set_image("nope", "a.png")
    assert excinfo.value.status_code == 404
