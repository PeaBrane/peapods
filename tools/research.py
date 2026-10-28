"""Snapshot private research or restore it without overwriting local work."""

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = "PeaBrane/peapods-research"
ROOTS = ("notes", "refs", "paper")
SMALL_FILE_LIMIT = 1024 * 1024
SKIP_DIRS = {
    ".claude",
    ".codex",
    ".git",
    ".idea",
    ".ipynb_checkpoints",
    ".lake",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "node_modules",
    "target",
    "venv",
}
SOURCE_SUFFIXES = {
    ".bib",
    ".c",
    ".cc",
    ".cjs",
    ".cls",
    ".cpp",
    ".h",
    ".hpp",
    ".ini",
    ".ipynb",
    ".jl",
    ".js",
    ".lean",
    ".md",
    ".mjs",
    ".py",
    ".pyi",
    ".r",
    ".rs",
    ".rst",
    ".sh",
    ".sty",
    ".tex",
    ".toml",
    ".ts",
    ".yaml",
    ".yml",
}
MEDIA_SUFFIXES = {".jpeg", ".jpg", ".pdf", ".png", ".svg", ".webp"}
BULK_SUFFIXES = {
    ".bin",
    ".bz2",
    ".dump",
    ".gz",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".tar",
    ".tgz",
    ".wit",
    ".xz",
    ".zip",
    ".zst",
}
SECRET_SUFFIXES = {".key", ".kdbx", ".p12", ".pem", ".pfx"}


def run(*command: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"Command failed: {command[0]}")
    return result.stdout.strip()


def tracked_paths(root: Path) -> set[Path]:
    output = run("git", "ls-files", "-z", "--", *ROOTS, cwd=root)
    return {Path(name) for name in output.split("\0") if name}


def include_file(source: Path) -> bool:
    if source.name.startswith(".") and source.name != ".python-version":
        return False
    suffix = source.suffix.lower()
    if suffix in SECRET_SUFFIXES:
        return False
    if suffix in SOURCE_SUFFIXES | MEDIA_SUFFIXES:
        return True
    if suffix in BULK_SUFFIXES or source.stat().st_size > SMALL_FILE_LIMIT:
        return False
    with source.open("rb") as handle:
        sample = handle.read(8192)
    if b"\0" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def research_files(root: Path, *, snapshot: bool) -> list[Path]:
    files = []
    for name in ROOTS:
        base = root / name
        if base.is_symlink():
            raise ValueError(f"Research root is a symlink: {base}")
        if not base.exists():
            continue
        if not base.is_dir():
            raise ValueError(f"Research root is not a directory: {base}")
        for directory, dirs, names in os.walk(base, followlinks=False):
            parent = Path(directory)
            for child in [*dirs, *names]:
                if (parent / child).is_symlink() and not snapshot:
                    raise ValueError(f"Snapshot contains a symlink: {parent / child}")
            if snapshot:
                dirs[:] = [
                    child
                    for child in dirs
                    if child not in SKIP_DIRS and not (parent / child).is_symlink()
                ]
            elif any(child in SKIP_DIRS for child in dirs):
                raise ValueError(f"Snapshot contains a managed directory: {parent}")
            for child in names:
                source = parent / child
                if source.is_symlink():
                    continue
                if not source.is_file():
                    raise ValueError(f"Not a regular file: {source}")
                if snapshot and not include_file(source):
                    continue
                files.append(source.relative_to(root))
    return sorted(files)


def snapshot(
    root: Path, destination: Path, tracked: set[Path], commit: str
) -> tuple[int, int]:
    if destination.is_symlink():
        raise ValueError(f"Snapshot destination is a symlink: {destination}")
    destination = destination.resolve()
    if destination.is_relative_to(root.resolve()):
        raise ValueError("Snapshot destination must be outside the source checkout")
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError(f"Snapshot destination must be empty: {destination}")
    selected = [
        ref for ref in research_files(root, snapshot=True) if ref not in tracked
    ]
    destination.mkdir(exist_ok=True)
    total = 0
    for ref in selected:
        target = destination / ref
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / ref, target)
        total += target.stat().st_size
    metadata = {
        "format": 1,
        "source_repository": "PeaBrane/peapods",
        "source_commit": commit,
        "files": len(selected),
        "bytes": total,
    }
    (destination / "snapshot.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / ".gitignore").write_text(".DS_Store\n.venv/\n__pycache__/\n")
    (destination / "README.md").write_text(
        "# Peapods private research\n\n"
        "Lightweight notes, papers, references, ledger, source, figures, and small evidence files.\n"
        "Bulk datasets, witnesses, dumps, archives, environments, and binaries are omitted.\n"
        "Omitted artifacts remain on the source Mac; no retrieval manifest is required.\n\n"
        "From a peapods checkout, restore with `.venv/bin/python tools/research.py restore`.\n"
        "Existing files must match or restoration stops before copying. The temporary clone is removed.\n"
        "The ledger reports absent computational artifacts as informational `UNAVAILABLE`; recorded\n"
        "mathematical status and earlier verification history are preserved.\n\n"
        "To prepare a new snapshot, run `.venv/bin/python tools/research.py snapshot <empty-directory>`\n"
        "from the source checkout. Review it before updating this private repository.\n"
    )
    return len(selected), total


def restore(
    root: Path, source: Path, tracked: set[Path], *, dry_run: bool = False
) -> tuple[int, int]:
    metadata_file = source / "snapshot.json"
    if metadata_file.is_symlink():
        raise ValueError("Snapshot metadata is a symlink")
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        if (
            not isinstance(metadata, dict)
            or metadata.get("format") != 1
            or metadata.get("source_repository") != "PeaBrane/peapods"
        ):
            raise ValueError("Not a supported peapods research snapshot")
    incoming = research_files(source, snapshot=False)
    if not incoming:
        raise ValueError("Snapshot contains no research files")
    pending = []
    conflicts = []
    for ref in incoming:
        target = root / ref
        if ref in tracked:
            conflicts.append(f"public tracked file: {ref}")
            continue
        blocked = False
        for component in [target, *target.parents]:
            if component == root:
                break
            if component.is_symlink() or (
                component != target and component.exists() and not component.is_dir()
            ):
                conflicts.append(f"unsafe destination: {ref}")
                blocked = True
                break
        if blocked:
            continue
        if target.exists():
            if not target.is_file() or not filecmp.cmp(
                source / ref, target, shallow=False
            ):
                conflicts.append(f"local file differs: {ref}")
            continue
        pending.append(ref)
    if conflicts:
        preview = "\n".join(conflicts[:10])
        raise ValueError(
            f"Restore blocked ({len(conflicts)} conflicts); nothing copied:\n{preview}"
        )
    if not dry_run:
        for ref in pending:
            target = root / ref
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / ref, target)
    return len(pending), len(incoming) - len(pending)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser(
        "snapshot", help="copy lightweight research to an empty directory"
    )
    export.add_argument("destination", type=Path)
    import_command = commands.add_parser(
        "restore", help="restore research from the private GitHub repo"
    )
    import_command.add_argument(
        "--dry-run",
        action="store_true",
        help="check access and conflicts without copying",
    )
    args = parser.parse_args()
    try:
        root = Path(
            run(
                "git",
                "rev-parse",
                "--show-toplevel",
                cwd=Path(__file__).resolve().parent.parent,
            )
        )
        tracked = tracked_paths(root)
        if args.command == "snapshot":
            commit = run("git", "rev-parse", "HEAD", cwd=root)
            count, size = snapshot(root, args.destination, tracked, commit)
            print(
                f"Snapshot: {count} files, {size / 1024**2:.1f} MiB in {args.destination.absolute()}"
            )
            return 0
        metadata = json.loads(run("gh", "api", f"repos/{REPOSITORY}"))
        if metadata.get("full_name") != REPOSITORY or not metadata.get("private"):
            raise ValueError(
                "Expected the private PeaBrane/peapods-research repository"
            )
        with tempfile.TemporaryDirectory(prefix="peapods-research-") as scratch:
            checkout = Path(scratch) / "research"
            run(
                "gh",
                "repo",
                "clone",
                REPOSITORY,
                str(checkout),
                "--",
                "--depth",
                "1",
                "--quiet",
            )
            count, unchanged = restore(root, checkout, tracked, dry_run=args.dry_run)
        action = "Would restore" if args.dry_run else "Restored"
        print(
            f"{action} {count} files; {unchanged} already identical. Temporary clone removed."
        )
        return 0
    except (OSError, ValueError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
