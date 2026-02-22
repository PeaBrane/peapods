# Project: Peapods

- Only refactor when new changes would introduce duplicate code; keep refactors light, and ask the user before doing heavy or multi-file ones.
- Never speculate about APIs or code behavior; explore definitions or ask the user when unsure.
- Do not modify code when the user is only asking questions; only change code on explicit triggers like "can you" or "ship it."
- Only run lint at the very end; if lint errors look stale, re-run once or leave them.
- Never run `cargo check`, `cargo build`, or `cargo test` after Rust changes; the user will handle compilation.
- Prefer structured and concise error messages.
- Favor denested control flow when possible (e.g., use let else or early returns instead of nested conditionals).
- All module imports should be placed at the top of the file; avoid inline imports.
- Do not add comments to code that is self-explanatory.
- When running Python commands, always use the project's virtual environment: `.venv/bin/python` or `.venv/bin/pip`.
