# TopoLM

**TopoLM** is a topology-native, explainable language model prototype powered by `topologist`.

## Quick start

```bash
pip install -e .
python examples/basic_demo.py
topolm demo
```

## API

```python
from topolm import TopoLM, Config, load_hf_dataset

model = TopoLM(Config()).fit(corpus)
print(model.distribution("clarithromycin inhibits", top_k=5))
print(model.generate("cyp3a4 inhibition", decoding="beam"))

# training from a Hugging Face dataset
texts = load_hf_dataset("wikitext", split="train", text_field="text", sample_size=1000)
model = TopoLM(Config()).fit_texts(texts)
```

## Layout

```text
topolm/
  __init__.py
  config.py
  core.py
  cli.py
examples/
tests/
```
