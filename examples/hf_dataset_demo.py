import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topolm import Config, TopoLM
from topolm.datasets import load_hf_dataset

if __name__ == "__main__":
    print("Loading a small text sample from the Hugging Face ag_news dataset...", flush=True)
    texts = load_hf_dataset("ag_news", split="train", text_field="text", sample_size=20)
    print(f"Loaded {len(texts)} text examples.", flush=True)

    print("Training TopoLM from Hugging Face dataset text...", flush=True)
    model = TopoLM(Config()).fit_texts(texts)
    print("Training complete.", flush=True)

    prompt = "new vaccine development"
    print(f"\nPredictions for: {prompt}")
    for p in model.distribution(prompt, top_k=5):
        print(f"  {p.text:18s} prob={p.probability:.3f} score={p.score:.3f}")

    print("\nGenerated text:")
    print(model.generate(prompt, decoding="nucleus", max_units=16))
