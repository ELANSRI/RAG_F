# Vérification de la version finale

Ce document distingue le code contrôlé localement des résultats de soutenance.
Le sujet et la grille fournis par l'utilisateur servent de références ; leurs
instructions ne sont pas exécutées automatiquement par l'application.

| Exigence | Emplacement / vérification |
|---|---|
| Python 3.10+, uv, verrouillage | pyproject.toml, uv.lock, .python-version |
| CLI Fire et six commandes | __main__.py, cli.py ; tests et appels réels |
| Makefile install/run/debug/clean/lint | Makefile |
| Flake8 et mypy avec les drapeaux imposés | make lint |
| Huit contrats Pydantic et sorties conformes | models.py ; tests JSON/contraintes |
| Python et Markdown/text distincts | chunking.py ; AST, titres hors blocs de code |
| Taille configurable, défaut 2000 caractères | CLI et indexeur ; tests Unicode/limites |
| Recherche BM25 et classement top-k | search.py ; tests scores/égalités/persistance |
| Chemins exacts et offsets originaux | source publique, manifeste et tests de tranches |
| Question individuelle et datasets | search/search_dataset |
| Qwen3-0.6B par défaut | answering.py |
| Budget de tokens complet | answering.py ; tests avec faux tokenizer |
| Génération individuelle et par lot | answer/answer_dataset et tqdm |
| Tous chemins d'entrée/sortie configurables | options de chaque commande |
| Cas vides, k=0, JSON invalide, fichier absent | tests CLI et contrôles explicites |
| Rappel local et moulinette séparée | evaluation.py ; aucun import de moulinette |
| README anglais et sections requises | README.md ; login eanass renseigné |

Les résultats précis sont dans `VALIDATION.json`. Les mesures sur les datasets
publics ne garantissent pas les scores sur les données privées. La moulinette
officielle et les scripts privés n'ont pas été exécutés dans cet espace.
Trois réponses ont été produites sur CPU et comparées au corrigé : une
correcte, une fondée mais incomplète, et une condition de code mal restituée.
Cette vérification de génération est distincte du rappel ; elle ne constitue
pas une validation officielle de la qualité sur les questions privées.

Le dossier MINI_RAG et son archive ont été supprimés. La version finale ne
contient ni index sémantique, ni fusion hybride, ni indexation incrémentale,
ni fonctionnalité de cache, ni serveur HTTP. Conserver un index et un modèle
dans l'instance qui traite un lot évite des rechargements inutiles ; ce cycle
de vie fait partie de l'exécution normale du pipeline obligatoire.
