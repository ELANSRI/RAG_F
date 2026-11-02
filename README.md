*This project has been created as part of the 42 curriculum by eanass.*

# RAG against the machine — mandatory version

## Description

This project answers questions about the supplied vLLM repository. It reads
Python and Markdown/text documents, splits them into bounded source spans,
retrieves evidence with BM25 and generates answers locally with
`Qwen/Qwen3-0.6B`. Pydantic validates the data exchanged between stages and
the JSON outputs. This version implements the mandatory part only.

## Instructions

Use Python 3.10 or later and uv. The development interpreter is Python 3.12.
Run all commands from the repository root. Dependencies are locked in
`uv.lock`; Linux uses CPU PyTorch packages.

```bash
uv sync
```

Prepare the supplied attachments in this layout:

```text
data/
  raw/vllm-0.10.1/          # Extracted source repository
  datasets/
    UnansweredQuestions/  # dataset_docs_public.json, dataset_code_public.json
    AnsweredQuestions/    # Reference datasets for local evaluation
```

Then run:

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "What is VLLM_RPC_TIMEOUT?" --k 5
uv run python -m src answer "What is VLLM_RPC_TIMEOUT?" --k 5 --device cpu
```

The first answer may download the default model. Later calls reuse the model
files on disk. Allow enough space for the dependencies and model weights.

This version writes index format 3. Rebuild any index produced by an earlier
version before searching. Source files must stay unchanged after indexing,
because the index stores their paths and character offsets.

## Example usage

All six required commands are available:

```bash
uv run python -m src index --raw_dir data/raw --index_dir data/processed --max_chunk_size 2000

uv run python -m src search "How do I load a LoRA adapter?" --k 5 --output_path data/output/single/search.json

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src answer "What is VLLM_RPC_TIMEOUT?" --k 5 \
  --max_context_tokens 3000 --max_new_tokens 256 --device cpu

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions \
  --device cpu

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --output_path data/output/evaluation/docs.json
```

Repeat the dataset commands with `dataset_code_public.json` for code questions.
Keep dataset scopes in output directory names to avoid overwriting files with
the same basename. `answer_dataset` uses the supplied search results; it does
not repeat retrieval and never receives the reference answers.

Single-query `search` and `answer` always save JSON. Their default files are
`data/output/single/search.json` and `data/output/single/answer.json`.
`--output_path` changes the destination. `--as_json True` prints the envelope
on stdout; save messages use stderr. Existing output files are replaced.

Use `--index_dir` on search commands to select another index. Chunk sizes must
be between 1 and 2000 characters. `k=0` is accepted and returns no sources;
answering without sources returns an explicit insufficient-evidence message.
Empty questions, negative k, missing files and malformed JSON produce readable
errors without an unhandled traceback.

## System architecture

```text
Documents → structure-aware chunks → BM25 index
Question → lexical terms → ranked references → measured prompt → Qwen → JSON
```

| File | Responsibility |
|---|---|
| `src/__main__.py` | Python Fire entry point |
| `src/cli.py` | Six commands, validation and readable errors |
| `src/models.py` | Eight required Pydantic contracts and internal records |
| `src/chunking.py` | Discovery, original text and Python/Markdown segmentation |
| `src/indexing.py` | Build and publish a complete index |
| `src/search.py` | Lexical terms, sparse BM25 and deterministic top-k |
| `src/answering.py` | Token-budgeted prompts and local generation |
| `src/evaluation.py` | Local source recall with interval overlap |
| `src/storage.py` | Validated JSON reads and atomic single-file writes |

`manifest.json` holds ordered passage coordinates, corpus location and chunk
size. `lexical.npz` holds BM25 weights; `vocabulary.json` maps terms to columns.
The manifest is published last. Concurrent rebuilding and searching is not
supported.

## Chunking strategy

Python uses AST line numbers to retain definitions, decorators and preceding
comments. Large classes split into methods; invalid syntax falls back to text.
Markdown uses headings outside fenced code blocks and builds heading paths.
Long spans split at paragraphs, lines and spaces, then fixed-width slices.
Every returned source is at most 2000 characters. Offsets reference the original
decoded text, with an inclusive start and exclusive end. They count Python
characters, not UTF-8 bytes or model tokens.

Smaller chunks reduce unrelated context but may separate a definition from its
explanation. Larger chunks retain more context and may introduce noise. The
default remains 2000. Effects of smaller sizes are recorded after measurement,
rather than assumed to improve retrieval.

## Retrieval method

BM25 uses `k1=1.2` and `b=0.75`. Indexing includes each passage's file path,
structural breadcrumb and original text. Compound identifiers contribute their
whole form and stemmed components. The tokenizer removes English stopwords;
it targets the English vLLM corpus and its identifiers.

Contributions are prepared in a sparse matrix. Construction uses CSR rows;
search uses CSC columns and a matrix-vector product. Positive finite scores
are ranked by decreasing score, with passage position resolving ties.
Partial selection preserves the ordering of a full sort.

## Answer generation

The default is `Qwen/Qwen3-0.6B`. The prompt contains numbered excerpts and
instructions to preserve exact identifiers, values, units and return structures.
The chat template disables thinking. The full formatted prompt is tokenized
and measured, reserving the maximum output length within the model's configured
window. Excerpts may be shortened; their prompt headers show the included span.

Generation uses greedy decoding and returns only new tokens. CPU inference
loads float32 weights. The model is loaded once per answering command and
shared by its questions. Progress bars
report indexing, batch search and batch answering. Retrieval never loads Qwen.
The public JSON retains retrieved candidate references; the prompt may contain
only a fitting subset or prefix. Generated citations are not automatically
verified for factual support.

## Performance analysis

Current local results are recorded in `docs/VALIDATION.json`. On the supplied
public datasets, the mandatory refactor preserves every ordered top-10 source
reference from the previous version across 199 questions:

| Dataset | Questions | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|
| Docs | 100 | 0.650 | 0.850 | 0.890 | 0.930 |
| Code | 99 | 0.586 | 0.818 | 0.869 | 0.919 |

Full indexing without application caches took about 43 seconds on this local
Mac, producing 25,298 passages from 1,847 contributing files out of 1,969
discovered files. A separate two-process CLI run searched 200 questions in
8.61 seconds, including startup and JSON writes; one public code question was
repeated with a fresh ID because that public dataset contains 99 questions.
These measurements do not establish timing on campus CPUs
or recall on private datasets. The local evaluator uses same-file IoU >=0.05
and averages source recall per question, including missing questions as zero.

The official moulinette and private exam scripts remain separate. The
application never imports or calls moulinette packages.

Three public questions were also answered with Qwen on CPU. The endpoint
answer was correct; the MoE answer was grounded but omitted part of the
parallelism description; the CUDA-graph answer still missed a necessary
condition. The stored local review documents that limitation rather than
claiming perfect generation or a guaranteed private-exam result.

## Design decisions and challenges

Keep one readable module per stage and preserve the known lexical behavior
while removing optional features. Store original offsets so the retrieved
evidence can be checked in the corpus. Validate malformed reference entries
before union selection can silently discard their answer fields. Keep public
field names unchanged and reject duplicate IDs or oversized outputs.

Exact prompt measurement avoids relying on a characters-per-token estimate.
Atomic JSON replacement avoids partially published output. Publishing the
manifest last marks index completeness; it does not provide a transaction
across all index files. A model response can remain incomplete or incorrect,
so generation quality must also be reviewed against its source excerpts.

## Development and understanding

```bash
make lint
make test
```

The Makefile also provides `install`, `run`, `debug` and `clean`.
`make run` defaults to indexing; select another command with
`ARGS='search "LoRA adapters" --k 5'`. Tests are development aids, not graded
deliverables. Run them in this workspace; the submission archive excludes
development tests and generated data.

Read the compact French walkthrough in `docs/COMPRENDRE.md` and the local
verification checklist in `docs/MANDATORY.md`.

## Resources and AI use

- The supplied RAG against the machine subject, version 2.0, and correction scale.
- Python AST: https://docs.python.org/3/library/ast.html
- Pydantic: https://docs.pydantic.dev/latest/
- Python Fire: https://google.github.io/python-fire/guide/
- SciPy sparse matrices: https://docs.scipy.org/doc/scipy/reference/sparse.html
- Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B
- BM25: https://doi.org/10.1561/1500000019
- uv: https://docs.astral.sh/uv/

AI assisted the refactor, removal of optional features, French explanations,
test adaptation and local verification. Retrieval behavior was compared with
the previous version on all supplied public questions. Generated answers were
reviewed separately; understanding and defending the implementation remain
the student's responsibility.
