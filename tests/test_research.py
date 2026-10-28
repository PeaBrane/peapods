"""Protect local work while transferring lightweight research snapshots."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from tools import research


def write(root, name, content="source\n"):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target


def make_snapshot(root):
    root.mkdir()
    write(
        root,
        "snapshot.json",
        json.dumps({"format": 1, "source_repository": "PeaBrane/peapods"}),
    )
    return root


def test_snapshot_preserves_core_and_small_evidence_but_omits_bulk(tmp_path):
    root = tmp_path / "project"
    keep = [
        "notes/proof.md",
        "notes/build/verify.cpp",
        "notes/manifest.json",
        "notes/audit.out",
        "notes/ledger/ledger.toml",
        "refs/paper.pdf",
        "paper/figure.png",
    ]
    for name in keep:
        write(root, name)
    for name in [
        "notes/vector.wit.xz",
        "notes/data/run.npz",
        "notes/.venv/code.py",
        "notes/target/source.rs",
        "notes/.git/config",
        "notes/.env",
        "notes/secret.pem",
        "paper/public.md",
    ]:
        write(root, name)
    write(root, "notes/large.json", "x" * (research.SMALL_FILE_LIMIT + 1))
    binary = root / "notes/verify"
    binary.write_bytes(b"ELF\0binary")
    destination = tmp_path / "snapshot"
    count, _ = research.snapshot(
        root, destination, {Path("paper/public.md")}, "test-commit"
    )
    assert count == len(keep)
    assert all((destination / name).exists() for name in keep)
    assert not (destination / "notes/vector.wit.xz").exists()
    assert not (destination / "notes/large.json").exists()
    assert not (destination / "notes/verify").exists()
    metadata = json.loads((destination / "snapshot.json").read_text())
    assert metadata["source_commit"] == "test-commit"


def test_snapshot_refuses_nonempty_destination(tmp_path):
    root = tmp_path / "project"
    write(root, "notes/proof.md")
    destination = tmp_path / "snapshot"
    write(destination, "precious.txt", "keep")
    with pytest.raises(ValueError, match="must be empty"):
        research.snapshot(root, destination, set(), "test")
    assert (destination / "precious.txt").read_text() == "keep"


def test_restore_is_idempotent_and_leaves_existing_artifacts(tmp_path):
    source = make_snapshot(tmp_path / "snapshot")
    write(source, "notes/proof.md")
    write(source, "refs/paper.pdf")
    root = tmp_path / "project"
    write(root, "notes/data/run.npz", "local data")
    assert research.restore(root, source, set()) == (2, 0)
    assert research.restore(root, source, set()) == (0, 2)
    assert (root / "notes/data/run.npz").read_text() == "local data"


def test_restore_preflights_all_conflicts_before_copying(tmp_path):
    source = make_snapshot(tmp_path / "snapshot")
    write(source, "notes/a-new.md")
    write(source, "notes/z-conflict.md", "remote")
    root = tmp_path / "project"
    write(root, "notes/z-conflict.md", "local")
    with pytest.raises(ValueError, match="nothing copied"):
        research.restore(root, source, set())
    assert not (root / "notes/a-new.md").exists()
    assert (root / "notes/z-conflict.md").read_text() == "local"


def test_restore_dry_run_does_not_create_research_files(tmp_path):
    source = make_snapshot(tmp_path / "snapshot")
    write(source, "notes/proof.md")
    root = tmp_path / "project"
    root.mkdir()
    assert research.restore(root, source, set(), dry_run=True) == (1, 0)
    assert not (root / "notes").exists()


def test_restore_does_not_require_a_manifest(tmp_path):
    source = tmp_path / "snapshot"
    write(source, "notes/proof.md")
    root = tmp_path / "project"
    root.mkdir()
    assert research.restore(root, source, set()) == (1, 0)


def test_restore_refuses_public_tracked_files(tmp_path):
    source = make_snapshot(tmp_path / "snapshot")
    write(source, "paper/public.md")
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(ValueError, match="public tracked file"):
        research.restore(root, source, {Path("paper/public.md")})
    assert not (root / "paper").exists()


@pytest.mark.parametrize("location", ["source", "destination", "parent"])
def test_restore_refuses_symlinks(tmp_path, location):
    source = make_snapshot(tmp_path / "snapshot")
    write(source, "notes/proof.md")
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    write(outside, "proof.md", "keep")
    if location == "source":
        (source / "notes/proof.md").unlink()
        (source / "notes/proof.md").symlink_to(outside / "proof.md")
    elif location == "destination":
        (root / "notes").mkdir()
        (root / "notes/proof.md").symlink_to(outside / "proof.md")
    else:
        (root / "notes").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|unsafe destination"):
        research.restore(root, source, set())
    assert (outside / "proof.md").read_text() == "keep"


def test_cli_clones_checks_copies_and_removes_temporary_clone(tmp_path, monkeypatch):
    source = make_snapshot(tmp_path / "snapshot")
    write(source, "notes/proof.md")
    root = tmp_path / "project"
    root.mkdir()
    clones = []

    def run(*command, cwd=None):
        if command[:3] == ("git", "rev-parse", "--show-toplevel"):
            return str(root)
        if command[:2] == ("git", "ls-files"):
            return ""
        if command[:2] == ("gh", "api"):
            return json.dumps({"full_name": research.REPOSITORY, "private": True})
        if command[:3] == ("gh", "repo", "clone"):
            destination = Path(command[4])
            clones.append(destination)
            shutil.copytree(source, destination)
            return ""
        raise AssertionError(command)

    monkeypatch.setattr(research, "run", run)
    monkeypatch.setattr(sys, "argv", ["research.py", "restore"])
    assert research.main() == 0
    assert (root / "notes/proof.md").read_text() == "source\n"
    assert len(clones) == 1
    assert not clones[0].parent.exists()


@pytest.mark.parametrize(
    "metadata", [{"full_name": research.REPOSITORY, "private": False}, None]
)
def test_cli_requires_existing_access_to_private_repository(
    tmp_path, monkeypatch, metadata
):
    root = tmp_path / "project"
    root.mkdir()

    def run(*command, cwd=None):
        if command[:3] == ("git", "rev-parse", "--show-toplevel"):
            return str(root)
        if command[:2] == ("git", "ls-files"):
            return ""
        if command[:2] == ("gh", "api"):
            if metadata is None:
                raise ValueError("Not Found (HTTP 404)")
            return json.dumps(metadata)
        raise AssertionError("Must not clone without access to the private repository")

    monkeypatch.setattr(research, "run", run)
    monkeypatch.setattr(sys, "argv", ["research.py", "restore"])
    assert research.main() == 1
    assert not (root / "notes").exists()
