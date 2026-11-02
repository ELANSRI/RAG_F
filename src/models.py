"""Contrats Pydantic obligatoires et petites structures internes de l'index."""

import uuid
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, model_validator


class MinimalSource(BaseModel):
    """Localiser un intervalle non vide de caractères dans un fichier."""

    file_path: str = Field(min_length=1)
    first_character_index: int = Field(ge=0, strict=True)
    last_character_index: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def check_interval(self) -> "MinimalSource":
        """Le début est inclus ; la fin est exclue et doit le dépasser."""
        if self.last_character_index <= self.first_character_index:
            raise ValueError("source end must be greater than its start")
        return self


class UnansweredQuestion(BaseModel):
    """Une question identifiée, sans réponse de référence."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Le corrigé d'une question, utilisé seulement pour l'évaluation."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Lire les questions sans masquer un corrigé invalide."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]

    @model_validator(mode="before")
    @classmethod
    def check_references(cls, payload: Any) -> Any:
        """Valider explicitement les entrées qui annoncent un corrigé."""
        if not isinstance(payload, dict):
            return payload
        entries = payload.get("rag_questions")
        if not isinstance(entries, list):
            return payload
        prepared = dict(payload)
        questions: list[Any] = []
        for entry in entries:
            if isinstance(entry, dict) and (
                "answer" in entry or "sources" in entry
            ):
                entry = AnsweredQuestion.model_validate(entry)
            questions.append(entry)
        prepared["rag_questions"] = questions
        return prepared

    @model_validator(mode="after")
    def check_identifiers(self) -> "RagDataset":
        """Un identifiant ne doit désigner qu'une question du dataset."""
        identifiers = [question.question_id for question in self.rag_questions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate question_id in dataset")
        return self


class MinimalSearchResults(BaseModel):
    """Les références classées pour une question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Ajouter le texte produit par Qwen au résultat de recherche."""

    answer: str


def check_results(results: Sequence[MinimalSearchResults], k: int) -> None:
    """Contrôler les identifiants, le nombre de sources et leurs largeurs."""
    identifiers = [result.question_id for result in results]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate question_id in search results")
    for result in results:
        if len(result.retrieved_sources) > k:
            raise ValueError("more sources than requested k")
        for source in result.retrieved_sources:
            width = source.last_character_index - source.first_character_index
            if width > 2000:
                raise ValueError("a retrieved source exceeds 2000 chars")


class StudentSearchResults(BaseModel):
    """Enveloppe JSON obligatoire des recherches."""

    search_results: list[MinimalSearchResults]
    k: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_results(self) -> "StudentSearchResults":
        """Appliquer les règles communes aux sorties publiques."""
        check_results(self.search_results, self.k)
        return self


class StudentSearchResultsAndAnswer(BaseModel):
    """Enveloppe JSON obligatoire des recherches accompagnées de réponses."""

    search_results: list[MinimalAnswer]
    k: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_results(self) -> "StudentSearchResultsAndAnswer":
        """Réutiliser les contrôles sans retirer les réponses."""
        check_results(self.search_results, self.k)
        return self


class Passage(MinimalSource):
    """Ajouter le chemin structurel utile à BM25, sans changer les bornes."""

    breadcrumb: str = ""

    def as_source(self) -> MinimalSource:
        """Exporter uniquement les trois champs du format public."""
        return MinimalSource(**self.model_dump(exclude={"breadcrumb"}))


class RankedPassage(BaseModel):
    """Un passage et son score BM25 interne."""

    passage: Passage
    relevance: float


class IndexManifest(BaseModel):
    """La ligne i de l'index doit toujours désigner le passage i."""

    format_version: Literal[3] = 3
    corpus_root: str
    max_chunk_size: int = Field(default=2000, ge=1, le=2000)
    passages: list[Passage]


class BuildSummary(BaseModel):
    """Compteurs de fichiers et durée d'une indexation complète."""

    discovered: int
    contributing: int
    unreadable: int
    passages: int
    elapsed: float
