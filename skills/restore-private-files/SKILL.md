---
name: restore-private-files
description: Restore Peapods private notes, references, manuscripts, and other ignored files into the selected checkout from its private companion repository. Use when setting up another machine or recovering missing private project files.
---

# Restore private files

Restore files from `PeaBrane/peapods-research` into the same relative paths in the selected
Peapods checkout. Resolve that checkout with `git rev-parse --show-toplevel` and run commands
from its root. The helper uses Git, an authenticated GitHub CLI (`gh`), and the checkout's
Python 3.11+ environment; it has no third-party Python dependencies.

Use existing GitHub access. Restoring files does not grant access or authorize installing or
authenticating tools. If a prerequisite is missing, report it. If this skill was installed
outside the checkout, locate [scripts/private_files.py](scripts/private_files.py) beside this
`SKILL.md` and pass its absolute path to the selected checkout's Python.

```sh
.venv/bin/python skills/restore-private-files/scripts/private_files.py restore
```

The helper verifies access to the private repository, clones it into temporary storage,
checks every destination before copying, and removes the temporary clone on completion or
failure. It restores `notes/`, `refs/`, and untracked `paper/` files. Existing identical files
are left alone. Differing local files, public tracked files, and symlinks block restoration
before any files are copied. Report conflicts without overwriting local work.

`restore --dry-run` previews which files would be copied; it also downloads the private
snapshot, so a separate preview repeats that download when followed by a real restore.

After restoring, read `notes/AGENTS.md` for the notes' own workflow. The ledger checker and
mathematical-status policy remain in `notes/ledger/`; this skill owns file transfer. Bulk
witnesses and datasets may be absent from a lightweight snapshot. Treat that as local
unavailability and leave those artifacts alone unless the user asks for them.

## Snapshot helper

The same bundled helper retains the existing export command for preparing an updated private
snapshot when requested:

```sh
.venv/bin/python skills/restore-private-files/scripts/private_files.py snapshot <empty-directory>
```

Use an empty destination outside the source checkout. The export keeps sources, PDFs, figures,
and small text evidence, and omits bulk computational formats, environments, caches, symlinks,
and files already tracked publicly. Other text-like files have a 1 MiB limit. Review exclusions
when important evidence uses an unusual format; these filters classify files by format and
size, not scientific importance. Exporting does not commit or publish the snapshot.
