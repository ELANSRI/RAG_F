"""Découper le corpus et publier un index BM25 complet."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from tqdm import tqdm

from .chunking import (
    list_documents,
    read_document,
    relative_name,
    segment_markdown,
    segment_plain_text,
    segment_python,
)
from .models import BuildSummary, IndexManifest, Passage
from .search import LexicalMatrix, lexical_terms
from .storage import write_model


def build_index(
    corpus_root: Path, destination: Path, max_chunk_size: int = 2000
) -> BuildSummary:
    """Préparer les fichiers avant de publier le manifeste en dernier.

    Args:
        corpus_root: Dossier des documents UTF-8 à lire récursivement.
        destination: Dossier dans lequel publier l'index complet.
        max_chunk_size: Largeur maximale en caractères, entre 1 et 2000.

    Returns:
        Les compteurs de fichiers/passages et la durée en secondes.
    """
    if type(max_chunk_size) is not int or not 1 <= max_chunk_size <= 2000:
        raise ValueError(
            "max_chunk_size must be an integer between 1 and 2000"
        )
    started = perf_counter()
    documents = list_documents(corpus_root)
    passages: list[Passage] = []
    searchable_terms: list[list[str]] = []
    unreadable = 0
    for document in tqdm(documents, desc="Indexing files", unit="file"):
        try:
            content = read_document(document)
        except (OSError, UnicodeDecodeError) as failure:
            tqdm.write(f"Skipping {document}: {failure}")
            unreadable += 1
            continue
        suffix = document.suffix.lower()
        if suffix == ".py":
            intervals = segment_python(content, max_chunk_size)
        elif suffix == ".md":
            intervals = segment_markdown(content, max_chunk_size)
        else:
            intervals = segment_plain_text(content, max_chunk_size)
        file_path = relative_name(document)
        for begin, end, breadcrumb in intervals:
            passages.append(Passage(
                file_path=file_path,
                first_character_index=begin,
                last_character_index=end,
                breadcrumb=breadcrumb,
            ))
            searchable_terms.append(lexical_terms(
                f"{file_path} {breadcrumb}\n{content[begin:end]}"
            ))
    if not passages:
        raise ValueError("no non-empty supported documents found")
    lexical = LexicalMatrix.from_documents(searchable_terms)
    manifest = IndexManifest(
        corpus_root=relative_name(corpus_root),
        max_chunk_size=max_chunk_size,
        passages=passages,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary)
        lexical.persist(staging)
        write_model(staging / "manifest.json", manifest)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "manifest.json").unlink(missing_ok=True)
        for item in staging.iterdir():
            if item.name != "manifest.json":
                item.replace(destination / item.name)
        (staging / "manifest.json").replace(destination / "manifest.json")
    return BuildSummary(
        discovered=len(documents),
        contributing=len({passage.file_path for passage in passages}),
        unreadable=unreadable,
        passages=len(passages),
        elapsed=perf_counter() - started,
    )
