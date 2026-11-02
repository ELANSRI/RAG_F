"""Six commandes obligatoires, chemins configurables et erreurs lisibles."""

import sys
from pathlib import Path

from .answering import DEFAULT_MODEL, LocalResponder, answer_results
from .evaluation import assess_retrieval
from .indexing import build_index
from .models import (
    MinimalSearchResults, RagDataset, StudentSearchResults, UnansweredQuestion,
)
from .search import SearchEngine
from .storage import read_model, write_model


def integer(value: object, name: str, minimum: int = 1) -> int:
    """Vérifier un entier sans accepter les booléens Python."""
    if (isinstance(value, bool) or not isinstance(value, int)
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def query_text(value: object) -> str:
    """Normaliser aussi les questions numériques reconnues par Fire."""
    wording = "" if value is None else str(value).strip()
    if not wording:
        raise ValueError("the query is empty")
    return wording


class Commands:
    """Indexer, rechercher, répondre et mesurer le rappel en BM25."""

    def index(self, max_chunk_size: int = 2000, raw_dir: str = "data/raw",
              index_dir: str = "data/processed") -> None:
        """Découper le corpus et enregistrer son index BM25."""
        summary = build_index(Path(raw_dir), Path(index_dir),
                              integer(max_chunk_size, "max_chunk_size"))
        print(f"Indexed {summary.passages} chunks from "
              f"{summary.contributing}/{summary.discovered} files "
              f"({summary.unreadable} unreadable) in {summary.elapsed:.2f}s")
        print(f"Ingestion complete! Indices saved under {index_dir}")

    def search(self, query: str, k: int = 10,
               index_dir: str = "data/processed", as_json: bool = False,
               output_path: str = "data/output/single/search.json") -> None:
        """Rechercher une question et toujours enregistrer son JSON public."""
        wording, requested = query_text(query), integer(k, "k", 0)
        matches = SearchEngine(Path(index_dir)).find(wording, requested)
        question = UnansweredQuestion(question=wording)
        result = MinimalSearchResults(
            question_id=question.question_id, question=wording,
            retrieved_sources=[hit.passage.as_source() for hit in matches],
        )
        envelope = StudentSearchResults(search_results=[result], k=requested)
        target = write_model(Path(output_path), envelope)
        if as_json:
            print(envelope.model_dump_json(indent=2))
        else:
            print(f"Top {len(matches)} sources for: {wording}")
            for rank, hit in enumerate(matches, 1):
                source = hit.passage
                print(f"{rank}. {source.file_path} "
                      f"[{source.first_character_index}:"
                      f"{source.last_character_index}] "
                      f"score={hit.relevance:.2f} ({source.breadcrumb})")
        print(f"Saved search results to {target}", file=sys.stderr)

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/search_results",
                       index_dir: str = "data/processed") -> None:
        """Chercher un lot sans utiliser ses éventuels corrigés."""
        requested = integer(k, "k", 0)
        dataset = read_model(Path(dataset_path), RagDataset)
        questions = [UnansweredQuestion(question_id=q.question_id,
                                        question=q.question)
                     for q in dataset.rag_questions]
        results = SearchEngine(Path(index_dir)).run_batch(questions, requested)
        target = write_model(Path(save_directory) / Path(dataset_path).name,
                             results)
        print(f"Saved student_search_results to {target}")

    def answer(self, query: str, k: int = 5,
               index_dir: str = "data/processed",
               model_name: str = DEFAULT_MODEL, max_context_tokens: int = 3000,
               max_new_tokens: int = 256, as_json: bool = False,
               output_path: str = "data/output/single/answer.json",
               device: str | None = None) -> None:
        """Rechercher les preuves, générer et enregistrer la réponse."""
        wording, requested = query_text(query), integer(k, "k", 0)
        question = UnansweredQuestion(question=wording)
        results = SearchEngine(Path(index_dir)).run_batch(
            [question], requested)
        responder = LocalResponder(
            model_name, integer(max_context_tokens, "max_context_tokens"),
            integer(max_new_tokens, "max_new_tokens"), device,
        )
        envelope = answer_results(results, responder)
        target = write_model(Path(output_path), envelope)
        if as_json:
            print(envelope.model_dump_json(indent=2))
        else:
            result = envelope.search_results[0]
            print(f"Question: {wording}\n\nAnswer: {result.answer}")
            print("\nSources:")
            for source in result.retrieved_sources:
                print(f"{source.file_path} [{source.first_character_index}:"
                      f"{source.last_character_index}]")
        print(f"Saved answers to {target}", file=sys.stderr)

    def answer_dataset(
        self, student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = DEFAULT_MODEL, max_context_tokens: int = 3000,
        max_new_tokens: int = 256, device: str | None = None,
    ) -> None:
        """Répondre aux recherches sauvegardées, avec un modèle par lot."""
        results = read_model(Path(student_search_results_path),
                             StudentSearchResults)
        responder = LocalResponder(
            model_name, integer(max_context_tokens, "max_context_tokens"),
            integer(max_new_tokens, "max_new_tokens"), device,
        )
        answers = answer_results(results, responder)
        target = write_model(
            Path(save_directory) / Path(student_search_results_path).name,
            answers,
        )
        print(f"Saved student_search_results_and_answer to {target}")

    def evaluate(self, student_search_results_path: str, dataset_path: str,
                 max_context_length: int = 2000,
                 output_path: str = "data/output/evaluation/report.json"
                 ) -> None:
        """Calculer le rappel local ; ne pas appeler la moulinette."""
        predictions = read_model(Path(student_search_results_path),
                                 StudentSearchResults)
        reference = read_model(Path(dataset_path), RagDataset)
        report = assess_retrieval(
            predictions, reference,
            integer(max_context_length, "max_context_length"),
        )
        write_model(Path(output_path), report)
        print(f"Local evaluation: {report.evaluated_questions} questions, "
              f"{report.missing_questions} missing")
        for k, recall in report.recall.items():
            print(f"Recall@{k}: {recall:.3f}")
        print("Run official validation separately with the moulinette.")
