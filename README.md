# TopoLM

<p align="center">
  A topology-native, explainable language model prototype powered by <a href="https://github.com/Arkay92/Topologist">topologist</a>.
</p>

<p align="center">
  <img width="256" height="256" alt="ChatGPT Image Jun 7, 2026, 11_38_36 AM" src="https://github.com/user-attachments/assets/969082e0-bb1c-4cda-9551-9cefdd23a06b" />
</p>

<p align="center">
  <a href="https://github.com/Arkay92/TopoLM/actions/workflows/publish.yml"><img alt="Publish" src="https://github.com/Arkay92/TopoLM/actions/workflows/publish.yml/badge.svg" /></a>
  <a href="https://pypi.org/project/topolm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/topolm.svg" /></a>
  <img alt="Python" src="https://img.shields.io/pypi/pyversions/topolm.svg" />
  <img alt="Downloads" src="https://img.shields.io/pypi/dm/topolm.svg" />
  <img alt="License" src="https://img.shields.io/pypi/l/topolm.svg" />
</p>

**TopoLM** combines:
- **Topology-native graph memory** using `topologist` and NetworkX.
- **Hyperdimensional encoding** for unit, domain, and sentence representations.
- **Evidence-based candidate retrieval** from phrase continuations, direct edges, and retrieved contexts.
- **Explainable scoring** with breakdowns of evidence, domain match, POS grammar, and repetition penalties.
- **Generation with multiple decoding strategies** (nucleus, beam, greedy) and phrase-tail detection.
- **Hugging Face dataset support** for training on large text corpora.
- **Persistence** with full state save/load, graph serialization, and memory reconstruction.

---

## Why Topology for Language Models?

Most neural LMs are **opaque black boxes**. Most symbolic systems are **brittle and limited**.

TopoLM sits between:

```
Input text
  -> Tokenize & domain detect
  -> Build symbolic graph (units, phrases, domains, POS)
  -> HDC encoding for each node
  -> Topological memory state
  
  -> Inference (next-token prediction, generation)
  -> Explainable evidence trails
  -> Drift detection & refinement
```

Each token, phrase, and domain relationship is stored **explicitly** in the graph, **encoded** into a high-dimensional bipolar vector, and **scored** by evidence, topology, and confidence. This gives you a language model that is:
- **Interpretable**: see exactly why a prediction was made.
- **Grounded**: graph structure prevents nonsense outputs.
- **Efficient**: no matrix multiplications; graph queries and HDC similarity.
- **Debuggable**: modify graph state, track provenance, refine confidence.

---

## Architecture

```
Text input
    |
    v
Tokenizer (unit, POS, domain, entity recognition)
    |
    v
Graph builder
  - Unit nodes (with frequency, domain, POS)
  - Phrase nodes (with multi-gram spans)
  - Domain nodes
  - Relations (next_unit, appears_near, likely_next, domain_related, has_pos)
    |
    v
HDC Memory (Topologist + fallback NetworkX)
  - Encode units, phrases, domains, positions into {-1,+1}^D vectors
  - Store graph topology
  - Bundled snapshots for drift
    |
    v
Inference (Predict or Generate)
  - Context Index (HDC similarity retrieval)
  - Candidate retrieval (phrase continuation, direct edges, domain priors, unigrams)
  - Evidence scoring (weighted by source: phrase, direct, RAG, domain, frequency)
  - Grammar validation (POS sequences)
  - Sampling (nucleus, beam, greedy)
```

---

## Install

```bash
pip install topolm
```

For Hugging Face dataset support:

```bash
pip install topolm[hf]
```

For development:

```bash
pip install -e ".[dev]"
pytest -q
python -m build
twine check dist/*
```

---

## Quick Start

### Basic Training and Prediction

```python
from topolm import TopoLM, Config

corpus = """
The cat sat on the mat.
The dog sat on the floor.
CYP3A4 inhibition increases drug exposure.
Clarithromycin inhibits CYP3A4.
"""

model = TopoLM(Config()).fit(corpus)

# Get next-token predictions
preds = model.distribution("clarithromycin inhibits", top_k=5)
for p in preds:
    print(f"  {p.text:20s} prob={p.probability:.3f} score={p.score:.3f}")

# Generate fluent text
generated = model.generate("cyp3a4 inhibition", decoding="beam")
print(generated)
```

### Training from Text List

```python
texts = [
    "Sentence one.",
    "Another sentence.",
    "Third sentence here.",
]
model = TopoLM(Config()).fit_texts(texts)
```

### Training from Hugging Face Dataset

```python
from topolm import load_hf_dataset

texts = load_hf_dataset(
    "wikitext",
    split="train",
    text_field="text",
    sample_size=1000
)
model = TopoLM(Config()).fit_texts(texts)
```

### Save and Load

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    path = model.save(tmpdir)
    loaded = TopoLM.load(path)
    print(loaded.distribution("clarithromycin inhibits", 3))
```

### Model Explanation

```python
explanation = model.explain("clarithromycin inhibits", "cyp3a4")
print(f"Score: {explanation['score']:.3f}")
print(f"Breakdown: {explanation['breakdown']}")
print(f"Evidence paths: {explanation['paths'][:3]}")
```

---

## CLI

Train and interact with a demo model:

```bash
topolm demo
```

Make predictions:

```bash
topolm predict "clarithromycin inhibits"
```

Generate text:

```bash
topolm generate "cyp3a4 inhibition" --decoding beam
```

---

## Main Features

### 1. **Hyperdimensional Unit Memory**

Tokens and phrases are encoded into stable bipolar vectors using seeded random generation:

```python
config = Config(dim=1024, seed=42)
hdc = HDC(dim=1024, seed=42)
vector = hdc.get("unit:clarithromycin")  # {-1, +1}^1024
```

### 2. **Symbolic Graph Topology**

Units, phrases, and domains are connected via typed relations:

- `next_unit`: direct token transitions
- `appears_near`: positional co-occurrence
- `likely_next`: phrase continuation
- `domain_related`: domain affinity
- `has_pos`: part-of-speech tagging

```python
g = model.graph
edges = list(g.out_edges("unit:clarithromycin", data=True))
for s, t, d in edges:
    print(f"{s} --{d['relation']}--> {t} (conf={d.get('confidence', 0.0):.2f})")
```

### 3. **Evidence-Based Candidate Retrieval**

Candidates are scored by multiple overlapping sources:

- **Phrase-based**: exact n-gram continuations from the graph
- **Direct edges**: observed next-token relations
- **Retrieved context**: HDC similarity to past sentences
- **Domain priors**: units from matching domain
- **Entity copy**: repeat entities from input
- **Frequency**: unigram statistics

```python
candidates = model.retrieve_candidates(
    units=["clarithromycin", "inhibits"],
    domain="drug_interaction",
    context_text="clarithromycin inhibits"
)
```

### 4. **Explainable Scoring**

Each prediction includes a breakdown:

```python
pred = model.distribution("clarithromycin inhibits", top_k=1)[0]
print(f"Text: {pred.text}")
print(f"Score: {pred.score:.3f}")
print(f"Probability: {pred.probability:.3f}")
print(f"Breakdown: {pred.breakdown}")
#  {'evidence': 0.5, 'phrase': 0.35, 'direct': 0.0, 'freq': 0.0, 'pos': 0.45, 'domain': 1.0, ...}
```

### 5. **Multiple Decoding Strategies**

Generate text using nucleus sampling, beam search, or greedy selection:

```python
# Nucleus sampling (default)
text = model.generate("prompt", decoding="nucleus", top_p=0.88)

# Beam search
text = model.generate("prompt", decoding="beam", beam_width=4)

# Greedy
text = model.generate("prompt", decoding="greedy")
```

### 6. **Domain Detection and Grounding**

Automatic domain detection prevents category confusion:

```python
domains = {
    "domestic": ["cat", "dog", "mat", "floor"],
    "cybersecurity": ["attacker", "exploit", "vulnerability"],
    "drug_interaction": ["cyp3a4", "clarithromycin", "inhibits"],
    "lm_research": ["language", "model", "topological"],
}
domain = model.tok.domain(["clarithromycin", "inhibits"])  # "drug_interaction"
```

### 7. **Full State Persistence**

Save and restore the complete model state, including graph and HDC memory:

```python
path = model.save("./model_checkpoint")
restored = TopoLM.load(path)
# Full parity: same predictions, same graph, same counts
```

### 8. **Graph Compaction**

Remove low-frequency edges to reduce memory:

```python
stats = model.mem.compact(min_edge_frequency=2)
print(f"Removed {stats['removed_edges']} edges")
```

---

## Configuration

Tune behavior via `Config`:

```python
from topolm import Config

config = Config(
    dim=1024,                      # HDC vector dimension
    seed=42,                       # Reproducibility
    window=8,                      # Co-occurrence window
    phrase_lengths=(2, 3, 4, 5),   # Phrase n-gram sizes
    max_candidates=96,             # Retrieval pool size
    inference_candidates=48,       # Top-k for scoring
    temperature=0.75,              # Softmax temperature
    default_top_p=0.88,            # Nucleus threshold
    default_beam_width=4,          # Beam search width
    fast_dev_mode=True,            # Disable slow features
)
model = TopoLM(config).fit(text)
```

---

## Examples

- [basic_demo.py](examples/basic_demo.py): Simple in-memory training and generation.
- [hf_dataset_demo.py](examples/hf_dataset_demo.py): Load and train on Hugging Face datasets.

---

## Project Structure

```
topolm/
  __init__.py          # Public API
  config.py            # Configuration dataclass
  core.py              # TopoLM, Memory, Tokenizer, HDC
  cli.py               # Command-line interface
  datasets.py          # Hugging Face dataset loaders
examples/
  basic_demo.py        # In-memory example
  hf_dataset_demo.py   # Hugging Face example
tests/
  test_smoke.py        # Smoke tests
.github/
  workflows/
    publish.yml        # PyPI publishing workflow
pyproject.toml         # Project metadata and dependencies
```

---

## Development

```bash
# Install with dev extras
pip install -e ".[dev]"

# Format and lint
ruff check .

# Run tests
pytest -q

# Build package
python -m build

# Check distributions
twine check dist/*
```

---

## Publishing

### PyPI Setup

1. Create a [PyPI account](https://pypi.org/account/register/).
2. Generate an [API token](https://pypi.org/manage/account/tokens/).
3. Store as a GitHub secret named `PYPI_API_TOKEN`.

### Publish via CI

Tag and push a release:

```bash
git tag v0.9.2
git push origin v0.9.2
```

The GitHub Actions workflow `.github/workflows/publish.yml` will automatically build and publish to PyPI.

### Manual Publishing

```bash
python -m build
twine upload dist/*
```

---

## Limitations and Future Work

- **No fine-tuning**: TopoLM learns from corpus statistics; no gradient-based learning.
- **Limited scalability**: Designed for interpretability at the cost of training speed.
- **Topologist dependency**: Requires `topologist>=0.4.0` for graph reasoning (fallback to NetworkX).
- **English-focused tokenization**: Custom regex tokenizer; non-English text may need adaptation.

Future improvements:
- Domain-specific confidence tuning.
- Multi-hop inference over learned relations.
- Tensor-backed HDC for GPU acceleration.
- Streaming/online updates.

---

## License

MIT

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) (if applicable) or open an issue.

---

## Citation

If you use TopoLM in research, please cite:

```bibtex
@software{topolm2024,
  title={TopoLM: A Topology-Native Explainable Language Model},
  author={McMenemy, Robert},
  url={https://github.com/Arkay92/TopoLM},
  year={2024},
  version={0.0.7},
}
```

---

## Acknowledgments

- [topologist](https://github.com/Arkay92/Topologist) for the hyperdimensional graph engine.
- [networkx](https://networkx.org/) for core graph algorithms.
- [huggingface/datasets](https://huggingface.co/docs/datasets/) for dataset loading.
