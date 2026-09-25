"""Guard the import direction between the repository's Python trees."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TREE_ROOTS = {
    "rcc": REPO_ROOT / "src" / "rcc",
    "experiments": REPO_ROOT / "experiments",
}
TREE_NAMES = frozenset(TREE_ROOTS)


def _module_name(path: Path, root: Path) -> str:
    """Return the import name for a file below one of the tree roots."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join((root.name, *parts))


def _absolute_imports(node: ast.AST, package: str) -> list[str]:
    """Resolve the import targets represented by one Import or ImportFrom node."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []

    if node.level == 0:
        base = node.module or ""
    else:
        parts = package.split(".") if package else []
        depth = len(parts) - node.level + 1
        if depth < 0:
            return []
        base = ".".join(parts[:depth])

    if not base:
        return []
    if node.module and node.module.split(".", 1)[0] in TREE_NAMES:
        return [base]
    if node.module:
        return [base]
    return [f"{base}.{alias.name}" for alias in node.names]


def _imports(path: Path, root: Path) -> list[tuple[int, str]]:
    """Return (line, target) pairs for import statements in one Python file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    module = _module_name(path, root)
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.extend((node.lineno, target) for target in _absolute_imports(node, package))
    return imports


def _tree_for(path: Path) -> tuple[str, Path] | None:
    """Return the owning tree and root for a path, if it is present."""
    for name, root in TREE_ROOTS.items():
        if root.is_dir() and path.is_relative_to(root):
            return name, root
    return None


def _is_forbidden(importer: str, target: str) -> bool:
    """Apply the allowed cross-tree edges to one resolved import target."""
    imported_tree = target.split(".", 1)[0]
    if imported_tree not in TREE_NAMES or imported_tree == importer:
        return False
    return importer == "rcc"


def test_source_tree_import_direction() -> None:
    """The library never imports the experiments tree; experiments may import the library."""
    violations: list[str] = []
    for path in sorted(
        path for root in TREE_ROOTS.values() if root.is_dir() for path in root.rglob("*.py")
    ):
        owner = _tree_for(path)
        assert owner is not None
        importer, root = owner
        try:
            imports = _imports(path, root)
        except SyntaxError as exc:
            violations.append(f"{path.relative_to(REPO_ROOT)}: cannot parse: {exc.msg}")
            continue
        for line, target in imports:
            if _is_forbidden(importer, target):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{line}: forbidden import '{target}'"
                )

    assert not violations, "forbidden cross-tree imports:\n" + "\n".join(violations)
