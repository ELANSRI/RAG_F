"""Point d'entrée : uv run python -m src <commande> [options]."""

import sys

import fire

from src.cli import Commands


def main() -> None:
    """Exposer les commandes et présenter leurs erreurs sans traceback."""
    try:
        fire.Fire(Commands)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as failure:
        print(f"Error: {failure}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
