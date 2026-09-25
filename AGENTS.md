# Agent instructions

Read [docs/development.md](docs/development.md) before editing.

- Keep the library focused on communication between existing agents.
- Keep transport selector-agnostic; CacheBack is the default callable in `selectors/`.
- Reuse the selector, capture and handoff code; do not add an agent framework.
- Every nontrivial change needs one meaningful runnable check and updated documentation.
- Keep comments under three lines. Put extended explanations in documentation.
- Keep full type hints and public summaries. Never relax type or complexity gates to pass.
- Run `.venv/bin/python scripts/check.py` before a commit. Never bypass hooks.
- Engine and cache changes must also pass the Transformers 4.57.1 compatibility check.
- Ask before adding dependencies, raising limits, changing measured claims or weakening tests.
- Review library changes for correctness and unnecessary abstraction before committing.
- Do not claim GPU, vLLM or benchmark validation from CPU tests alone.
- Keep experimental representations clearly marked as untested end to end.
- Keep this harness on `main`; preserve the frozen `paper` replication snapshot.
