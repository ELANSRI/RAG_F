"""Tokenisation, index BM25 creux et classement des passages."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import snowballstemmer
from scipy import sparse
from tqdm import tqdm

from .models import (
    IndexManifest,
    MinimalSearchResults,
    RankedPassage,
    StudentSearchResults,
    UnansweredQuestion,
)
from .storage import read_model

IGNORED_WORDS = frozenset("""
    a about above after again against all am an and any are aren as at be
    because been before being below between both but by can cannot could
    couldn did didn do does doesn doing don down during each few for from
    further had hadn has hasn have haven having he her here hers herself
    him himself his how i if in into is isn it its itself just let ll me
    more most mustn my myself no nor not now of off on once only or other
    ought our ours ourselves out over own re same shan she should shouldn
    so some such than that the their theirs them themselves then there
    these they this those through to too under until up us very was wasn
    we were weren what when where which while who whom why will with won
    would wouldn you your yours yourself yourselves ve
    """.split())
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_]+")
COMPONENT_PATTERN = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
ENGLISH_STEMMER = snowballstemmer.stemmer("english")


def expand_identifier(identifier: str) -> tuple[str, ...]:
    """Garder le nom composé et les racines de ses composants utiles."""
    stripped = identifier.strip("_")
    components = [
        part.lower()
        for section in stripped.split("_")
        for part in COMPONENT_PATTERN.findall(section)
    ]
    terms = [stripped.lower()] if len(components) > 1 else []
    terms.extend(
        str(ENGLISH_STEMMER.stemWord(part))
        for part in components
        if len(part) >= 2 and part not in IGNORED_WORDS
    )
    return tuple(terms)


def lexical_terms(content: str) -> list[str]:
    """Développer les identifiants en conservant toutes leurs fréquences."""
    terms: list[str] = []
    for identifier, count in Counter(
        IDENTIFIER_PATTERN.findall(content)
    ).items():
        terms.extend(expand_identifier(identifier) * count)
    return terms


class LexicalMatrix:
    """Lignes de passages, colonnes de termes et contributions BM25."""

    def __init__(
        self, coefficients: sparse.csc_matrix, term_columns: dict[str, int]
    ) -> None:
        """Associer les poids creux au dictionnaire terme-colonne."""
        self.coefficients = coefficients
        self.term_columns = term_columns

    @classmethod
    def from_documents(
        cls, documents: Iterable[Sequence[str]]
    ) -> "LexicalMatrix":
        """Précalculer les poids avec k1=1.2 et b=0.75."""
        lexicon: dict[str, int] = {}
        boundaries, columns, frequencies = [0], [], []
        for terms in documents:
            for term, count in Counter(terms).items():
                columns.append(lexicon.setdefault(term, len(lexicon)))
                frequencies.append(count)
            boundaries.append(len(columns))
        count = len(boundaries) - 1
        counts = sparse.csr_matrix(
            (
                np.asarray(frequencies, dtype=np.float32),
                np.asarray(columns, dtype=np.int32),
                np.asarray(boundaries, dtype=np.int64),
            ),
            shape=(count, max(1, len(lexicon))),
        )
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        average = max(float(lengths.mean()), 1e-9) if count else 1
        df = np.bincount(counts.indices, minlength=counts.shape[1])
        rarity = np.log1p((count - df + 0.5) / (df + 0.5)).astype(
            np.float32
        )
        penalty = 1.2 * (1 - 0.75 + 0.75 * lengths / average)
        rows = np.repeat(np.arange(count), np.diff(counts.indptr))
        weights = (
            rarity[counts.indices] * counts.data * (1.2 + 1)
            / (counts.data + penalty[rows])
        ).astype(np.float32)
        prepared = sparse.csr_matrix(
            (weights, counts.indices, counts.indptr), shape=counts.shape
        )
        return cls(sparse.csc_matrix(prepared), lexicon)

    def scores_for(self, query_terms: Sequence[str]) -> np.ndarray:
        """Additionner les colonnes utiles selon les fréquences de question."""
        repetitions = Counter(
            term for term in query_terms if term in self.term_columns
        )
        if not repetitions:
            return np.zeros(self.coefficients.shape[0], dtype=np.float32)
        columns = [self.term_columns[term] for term in repetitions]
        multipliers = np.asarray(list(repetitions.values()), dtype=np.float32)
        return np.asarray(
            self.coefficients[:, columns] @ multipliers, dtype=np.float32
        ).ravel()

    def persist(self, destination: Path) -> None:
        """Écrire les poids et le vocabulaire dans le dossier préparé."""
        destination.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(destination / "lexical.npz", self.coefficients)
        (destination / "vocabulary.json").write_text(
            json.dumps(self.term_columns), encoding="utf-8"
        )

    @classmethod
    def restore(cls, directory: Path) -> "LexicalMatrix":
        """Relire l'index et contrôler les numéros de colonnes."""
        coefficients = sparse.csc_matrix(
            sparse.load_npz(directory / "lexical.npz")
        )
        vocabulary = json.loads(
            (directory / "vocabulary.json").read_text(encoding="utf-8")
        )
        if not isinstance(vocabulary, dict) or any(
            not isinstance(term, str) or type(column) is not int
            for term, column in vocabulary.items()
        ):
            raise ValueError("invalid vocabulary; rebuild the index")
        if sorted(vocabulary.values()) != list(range(len(vocabulary))):
            raise ValueError("vocabulary columns are not contiguous")
        if coefficients.shape[1] != max(1, len(vocabulary)):
            raise ValueError("matrix and vocabulary dimensions differ")
        return cls(coefficients, vocabulary)


def best_positions(values: np.ndarray, requested: int) -> np.ndarray:
    """Garder les meilleurs scores finis positifs ; départager par position."""
    candidates = np.flatnonzero(np.isfinite(values) & (values > 0))
    if requested <= 0 or candidates.size == 0:
        return np.empty(0, dtype=np.int64)
    if candidates.size > requested:
        scores = values[candidates]
        boundary = scores.size - requested
        cutoff = np.partition(scores, boundary)[boundary]
        above = candidates[scores > cutoff]
        tied = candidates[scores == cutoff][:requested - above.size]
        candidates = np.concatenate((above, tied))
    order = np.lexsort((candidates, -values[candidates]))
    return np.asarray(candidates[order[:requested]], dtype=np.int64)


class SearchEngine:
    """Charger une fois l'index BM25 pour les questions d'une commande."""

    def __init__(self, directory: Path) -> None:
        """Relire le manifeste et vérifier son alignement avec les poids."""
        location = directory / "manifest.json"
        if not location.is_file():
            raise FileNotFoundError(f"no index in {directory}; run 'index'")
        self.manifest = read_model(location, IndexManifest)
        self.lexical = LexicalMatrix.restore(directory)
        if self.lexical.coefficients.shape[0] != len(self.manifest.passages):
            raise ValueError("passage table and lexical matrix are misaligned")

    def find(self, text: str, requested: int) -> list[RankedPassage]:
        """Retourner des références classées sans charger de modèle."""
        if requested <= 0 or not text.strip():
            return []
        scores = self.lexical.scores_for(lexical_terms(text))
        return [
            RankedPassage(
                passage=self.manifest.passages[position],
                relevance=float(scores[position]),
            )
            for position in best_positions(scores, requested)
        ]

    def run_batch(
        self, questions: Sequence[UnansweredQuestion], requested: int
    ) -> StudentSearchResults:
        """Chercher chaque question en conservant son identifiant public."""
        results = []
        for question in tqdm(questions, desc="Searching", unit="question"):
            matches = self.find(question.question, requested)
            results.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[
                    match.passage.as_source() for match in matches
                ],
            ))
        return StudentSearchResults(search_results=results, k=requested)
