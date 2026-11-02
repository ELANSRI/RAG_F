# Comprendre la version mandatory

Cette version conserve dix fichiers Python, dont le marqueur `__init__.py`.
Lis les fichiers dans cet ordre, en suivant une seule question concrète.

## 1. Entrer dans le programme

`__main__.py` donne la classe `Commands` à Python Fire. Une commande comme
`uv run python -m src search "LoRA" --k 5` appelle `Commands.search`.
`cli.py` vérifie les arguments, organise les étapes et sauvegarde le résultat.
Le `try/except` de `main` présente les erreurs sur stderr, sans traceback.

## 2. Définir les données

`models.py` contient les huit modèles demandés par le sujet : une référence,
des questions, un dataset, un résultat individuel et les enveloppes publiques.
Pydantic valide leurs champs et les règles entre champs. Une annotation Python
ordinaire ne suffit pas à contrôler une valeur pendant l'exécution.

Une source contient un chemin, un début inclus et une fin exclue.
`texte[10:30]` couvre vingt caractères. L'enveloppe publique refuse une source
de plus de 2000 caractères. Le validateur du dataset protège aussi les corrigés
incomplets contre un repli silencieux vers une question sans réponse.

## 3. Construire les passages

`chunking.py` lit les fichiers UTF-8. L'AST fournit les frontières Python sans
exécuter le code du corpus ; les titres fournissent les frontières Markdown.
Un morceau trop long est subdivisé sur les paragraphes, lignes et espaces.
Le dernier recours coupe à largeur fixe. Les bornes se rapportent toujours
au texte original lu par le programme. Le breadcrumb ajoute un chemin
structurel, comme `class Client > def send`, sans modifier ce texte.

## 4. Préparer BM25

`indexing.py` parcourt tout le corpus et construit la liste des passages.
Le chemin, le breadcrumb et le texte fournissent ensemble les termes indexés.
`search.py` reconnaît les identifiants, conserve les formes composées et leurs
composants, retire des stopwords anglais et calcule des racines lexicales.
Les répétitions sont conservées dans les fréquences.

BM25 combine rareté, fréquence et longueur :

```text
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
poids(t,d) = idf(t) × tf(t,d) × 2.2
             / (tf(t,d) + 1.2 × (0.25 + 0.75 × L(d)/L_moy))
```

`df` compte les passages contenant un terme ; `tf` compte ses occurrences
dans un passage. `L` compte les termes lexicaux, pas les caractères. Répéter
un terme dans le passage apporte un gain qui sature. La fréquence dans la
question multiplie sa contribution. Le score n'est pas une probabilité.

La matrice creuse a une ligne par passage et une colonne par terme.
CSR aide à la construire par lignes ; CSC aide à sélectionner les colonnes
d'une question. `manifest.json`, `lexical.npz` et `vocabulary.json` constituent
l'index persistant obligatoire. Ils ne contiennent pas les poids Qwen.

## 5. Chercher une question

`SearchEngine.find` applique la même tokenisation à la question. Les colonnes
de ses termes sont additionnées, pondérées par leurs fréquences :

```text
(N,U) @ (U,) → (N,)
```

Il y a un score par passage. `best_positions` conserve au plus k scores finis
et positifs ; les égalités sont départagées par position. La recherche produit
des références classées. Elle ne génère pas de réponse.

## 6. Faire répondre Qwen

`answering.py` relit les références et construit des extraits numérotés.
Le prompt contient les règles, les extraits et la question dans le format du
tokenizer Qwen. Si W est la capacité du modèle, G la sortie maximale, B la
taille sans extrait et C le budget de contexte :

```text
limite du prompt complet = min(W - G, B + C)
```

Les instructions et les en-têtes comptent. Un préfixe peut remplacer un
extrait trop long ; son en-tête affiche alors sa vraie borne de fin.
Qwen produit de nouveaux tokens, puis le code enlève les tokens du prompt
avant de décoder. Les sources vides donnent une réponse de preuve insuffisante.

Cette version n'effectue aucun entraînement ni calcul d'embeddings de recherche.
BM25 sélectionne le texte ; Qwen lit ce texte et génère une réponse. Les poids
préentraînés du modèle et l'index documentaire sont deux choses distinctes.

## 7. Sauvegarder et vérifier

`storage.py` valide le JSON lu et publie un JSON écrit par remplacement d'un
temporaire du même dossier. `search` et `answer` sauvegardent même sans
`--as_json True` : ce drapeau règle l'affichage.

`evaluation.py` compare les positions du même fichier exact par IoU :
intersection / union. Une source attendue est retrouvée si un des k résultats
atteint 0,05. Le rapport fait la moyenne des rappels des questions. Les
questions manquantes comptent zéro. Ce score mesure la recherche des sources,
pas la vérité de la réponse générée.

Pour chaque fonction, retrouve ses entrées, sa sortie, ses effets sur le disque
et ses erreurs. Puis vérifie ces quatre points avec une question réelle.
