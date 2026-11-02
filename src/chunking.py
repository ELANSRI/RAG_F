"""Lire et découper le corpus sans exécuter son code ni changer ses offsets."""

import ast
import os
import re
from pathlib import Path
from typing import Sequence

Interval = tuple[int, int]
LabeledInterval = tuple[int, int, str]
SUPPORTED_SUFFIXES = {".py", ".md", ".txt", ".rst"}
BREAK_MARKERS = ("\n\n", "\n", " ")
DEFINITION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
HEADING_PATTERN = re.compile(r"#{1,6}(?=\s)")
FENCE_PATTERN = re.compile(r"[ \t]*(`{3,}|~{3,})")


def list_documents(root: Path) -> list[Path]:
    """Énumérer les fichiers acceptés dans un ordre reproductible."""
    if not root.is_dir():
        raise FileNotFoundError(f"raw corpus directory not found: {root}")
    return sorted(
        document
        for document in root.rglob("*")
        if document.is_file() and document.suffix.lower() in SUPPORTED_SUFFIXES
    )


def relative_name(document: Path) -> str:
    """Exprimer le chemin POSIX relativement au dossier d'exécution."""
    return Path(os.path.relpath(document.resolve(), Path.cwd())).as_posix()


def read_document(document: Path) -> str:
    """Décoder en UTF-8 avec le traitement standard des fins de ligne."""
    return document.read_text(encoding="utf-8")


def line_offsets(content: str) -> list[int]:
    """Repérer chaque début de ligne et la fin du texte en caractères."""
    offsets = [0]
    for match in re.finditer("\n", content):
        offsets.append(match.end())
    if offsets[-1] != len(content):
        offsets.append(len(content))
    return offsets


def trim_interval(content: str, begin: int, stop: int) -> Interval | None:
    """Retirer les blancs aux deux extrémités, sans déplacer le texte."""
    while begin < stop and content[begin].isspace():
        begin += 1
    while stop > begin and content[stop - 1].isspace():
        stop -= 1
    return (begin, stop) if begin < stop else None


def separator_ends(
    content: str, begin: int, stop: int, separator: str
) -> list[int]:
    """Repérer les fins des séparateurs strictement internes à l'intervalle."""
    cuts: list[int] = []
    position = content.find(separator, begin, stop)
    while position != -1:
        cut = position + len(separator)
        if begin < cut < stop:
            cuts.append(cut)
        position = content.find(separator, cut, stop)
    return cuts


def pack_intervals(
    intervals: Sequence[Interval], character_limit: int
) -> list[Interval]:
    """Regrouper les fragments voisins tant que leur largeur tient."""
    packed: list[Interval] = []
    for begin, stop in intervals:
        if packed and stop - packed[-1][0] <= character_limit:
            packed[-1] = (packed[-1][0], stop)
        else:
            packed.append((begin, stop))
    return packed


def split_interval(
    content: str,
    begin: int,
    stop: int,
    character_limit: int,
    separators: Sequence[str] = BREAK_MARKERS,
) -> list[Interval]:
    """Couper par paragraphes, lignes, espaces, puis largeur fixe."""
    if character_limit <= 0:
        raise ValueError("character_limit must be positive")
    if stop - begin <= character_limit:
        return [(begin, stop)]
    for index, separator in enumerate(separators):
        cuts = separator_ends(content, begin, stop, separator)
        if not cuts:
            continue
        boundaries = [begin, *cuts, stop]
        fragments: list[Interval] = []
        for fragment_begin, fragment_stop in zip(boundaries, boundaries[1:]):
            fragments.extend(
                split_interval(
                    content, fragment_begin, fragment_stop, character_limit,
                    separators[index + 1:],
                )
            )
        return pack_intervals(fragments, character_limit)
    return [
        (position, min(position + character_limit, stop))
        for position in range(begin, stop, character_limit)
    ]


def finish_intervals(
    content: str, intervals: Sequence[LabeledInterval], character_limit: int
) -> list[LabeledInterval]:
    """Limiter les largeurs, retirer les blancs et ignorer les vides."""
    segments: list[LabeledInterval] = []
    for begin, stop, breadcrumb in intervals:
        for fragment_begin, fragment_stop in split_interval(
            content, begin, stop, character_limit
        ):
            trimmed = trim_interval(content, fragment_begin, fragment_stop)
            if trimmed is not None:
                segments.append((trimmed[0], trimmed[1], breadcrumb))
    return segments


def segment_plain_text(
    content: str, character_limit: int = 2000
) -> list[LabeledInterval]:
    """Découper le texte brut sans breadcrumb.

    Args:
        content: Texte original à découper, sans le réécrire.
        character_limit: Largeur maximale d'un passage en caractères.

    Returns:
        Liste de (début, fin exclusive, breadcrumb vide).
    """
    return finish_intervals(content, [(0, len(content), "")], character_limit)


def first_definition_line(statement: ast.stmt) -> int:
    """Trouver la première ligne d'une définition, décorateurs compris."""
    decorators: list[ast.expr] = getattr(statement, "decorator_list", [])
    return min([statement.lineno, *(item.lineno for item in decorators)])


def include_leading_comments(source_lines: Sequence[str], line: int) -> int:
    """Inclure les commentaires immédiatement placés avant une définition."""
    while line > 0 and source_lines[line - 1].lstrip().startswith("#"):
        line -= 1
    return line


def describe_definition(statement: ast.stmt) -> str:
    """Nommer une classe ou une fonction pour construire son breadcrumb."""
    if isinstance(statement, ast.ClassDef):
        return f"class {statement.name}"
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"def {statement.name}"
    return ""


def segment_python(
    content: str, character_limit: int = 2000
) -> list[LabeledInterval]:
    """Suivre l'AST, ou découper en texte brut si l'analyse échoue.

    Args:
        content: Code Python original, analysé sans être exécuté.
        character_limit: Largeur maximale d'un passage en caractères.

    Returns:
        Liste de (début, fin exclusive, breadcrumb de définition ou vide).
    """
    try:
        syntax_tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return segment_plain_text(content, character_limit)
    source_lines = content.splitlines(keepends=True)
    offsets = line_offsets(content)

    def line_position(line: int) -> int:
        """Convertir une frontière de ligne en position dans la chaîne."""
        return offsets[min(line, len(offsets) - 1)]

    boundaries: list[tuple[int, str]] = [(0, "")]

    def visit_definition(statement: ast.stmt, enclosing: str) -> None:
        """Ajouter les bornes du nœud et des membres d'une grande classe."""
        label = describe_definition(statement)
        breadcrumb = f"{enclosing} > {label}" if enclosing else label
        first = include_leading_comments(
            source_lines, first_definition_line(statement) - 1
        )
        end_line = statement.end_lineno or statement.lineno
        boundaries.append((line_position(first), breadcrumb))
        if (
            isinstance(statement, ast.ClassDef)
            and line_position(end_line) - line_position(first)
            > character_limit
        ):
            for child in statement.body:
                if isinstance(child, DEFINITION_NODES):
                    visit_definition(child, breadcrumb)
        boundaries.append((line_position(end_line), enclosing))

    for statement in syntax_tree.body:
        if isinstance(statement, DEFINITION_NODES):
            visit_definition(statement, "")
    boundaries.append((len(content), ""))
    boundaries.sort(key=lambda boundary: boundary[0])
    intervals: list[LabeledInterval] = []
    for (begin, breadcrumb), (stop, _) in zip(boundaries, boundaries[1:]):
        if stop > begin:
            intervals.append((begin, stop, breadcrumb))
    return finish_intervals(content, intervals, character_limit)


def segment_markdown(
    content: str, character_limit: int = 2000
) -> list[LabeledInterval]:
    """Suivre les titres hors blocs clôturés et conserver leur hiérarchie.

    Args:
        content: Texte Markdown original à découper sans le réécrire.
        character_limit: Largeur maximale d'un passage en caractères.

    Returns:
        Liste de (début, fin exclusive, breadcrumb des titres ou vide).
    """
    sections: list[LabeledInterval] = []
    heading_stack: list[tuple[int, str]] = []
    section_begin = 0
    breadcrumb = ""
    fence: str | None = None
    offsets = line_offsets(content)
    for line_begin, line_stop in zip(offsets, offsets[1:]):
        line = content[line_begin:line_stop]
        fence_match = FENCE_PATTERN.match(line)
        if fence is not None:
            if fence_match and fence_match.group(1).startswith(fence):
                fence = None
            continue
        if fence_match:
            fence = fence_match.group(1)
            continue
        heading = HEADING_PATTERN.match(line)
        if heading is None:
            continue
        sections.append((section_begin, line_begin, breadcrumb))
        level = len(heading.group(0))
        title = line[level:].strip().strip("#").strip()
        heading_stack = [item for item in heading_stack if item[0] < level] + [
            (level, title)
        ]
        breadcrumb = " > ".join(item[1] for item in heading_stack)
        section_begin = line_begin
    sections.append((section_begin, len(content), breadcrumb))
    return finish_intervals(content, sections, character_limit)
