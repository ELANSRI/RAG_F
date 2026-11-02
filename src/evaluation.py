"""Mesurer localement le rappel des sources ; la moulinette reste externe."""

from pydantic import BaseModel

from .models import (
    AnsweredQuestion, MinimalSource, RagDataset, StudentSearchResults,
)


class Assessment(BaseModel):
    """Moyenne des rappels par question à plusieurs valeurs de k."""

    evaluated_questions: int
    missing_questions: int
    recall: dict[int, float]


def interval_overlap(
    expected: MinimalSource, proposed: MinimalSource
) -> float:
    """Intersection sur union de deux intervalles du même fichier exact."""
    if expected.file_path != proposed.file_path:
        return 0.0
    intersection = max(0, min(expected.last_character_index,
                              proposed.last_character_index)
                       - max(expected.first_character_index,
                             proposed.first_character_index))
    union = (expected.last_character_index - expected.first_character_index
             + proposed.last_character_index - proposed.first_character_index
             - intersection)
    return intersection / union


def assess_retrieval(predictions: StudentSearchResults, reference: RagDataset,
                     source_limit: int = 2000) -> Assessment:
    """Une source est retrouvée si un résultat atteint une IoU >= 0,05."""
    if not reference.rag_questions:
        raise ValueError("reference dataset is empty")
    if any(not isinstance(q, AnsweredQuestion)
           for q in reference.rag_questions):
        raise ValueError("evaluation requires AnsweredQuestions references")
    for prediction in predictions.search_results:
        for source in prediction.retrieved_sources:
            width = source.last_character_index - source.first_character_index
            if width > source_limit:
                raise ValueError("retrieved source exceeds evaluation limit")
    by_id = {result.question_id: result
             for result in predictions.search_results}
    cutoffs = sorted(k for k in {1, 3, 5, 10, predictions.k}
                     if 0 < k <= predictions.k)
    totals = {k: 0.0 for k in cutoffs}
    missing = 0
    for question in reference.rag_questions:
        if not isinstance(question, AnsweredQuestion):
            continue
        if not question.sources:
            raise ValueError("a reference question has no expected sources")
        result = by_id.get(question.question_id)
        if result is None:
            missing += 1
            continue
        if result.question != question.question:
            raise ValueError(f"question text mismatch: {question.question_id}")
        for k in cutoffs:
            matched = sum(any(interval_overlap(source, hit) >= 0.05
                              for hit in result.retrieved_sources[:k])
                          for source in question.sources)
            totals[k] += matched / len(question.sources)
    return Assessment(evaluated_questions=len(reference.rag_questions),
                      missing_questions=missing,
                      recall={k: total / len(reference.rag_questions)
                              for k, total in totals.items()})
