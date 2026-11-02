"""Lecture validée et écriture d'un JSON complet, sans cache."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypeVar

from pydantic import BaseModel

Record = TypeVar("Record", bound=BaseModel)


def read_model(path: Path, model: type[Record]) -> Record:
    """Lire un fichier UTF-8 et valider ses données avec le modèle demandé."""
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def write_model(path: Path, record: BaseModel) -> Path:
    """Créer les parents puis publier le JSON par remplacement atomique."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=".rag-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(record.model_dump_json(indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
