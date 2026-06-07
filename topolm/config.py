from dataclasses import dataclass

PUNCT = {".", ",", ";", ":", "?", "!"}
BOUNDARY = {"<bos>", "<eos>"}
HUBS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "when", ".", ","}

@dataclass
class Config:
    dim: int = 1024
    seed: int = 42
    window: int = 8
    phrase_lengths: tuple[int, ...] = (2, 3, 4, 5)
    max_candidates: int = 96
    inference_candidates: int = 48
    prediction_cache_max: int = 4096
    temperature: float = 0.75
    max_runtime_seconds: float = 5.0
    default_top_p: float = 0.88
    default_beam_width: int = 4
    fast_dev_mode: bool = True
    max_reranker_sentences: int = 80
    negatives_per_positive: int = 2
    hub_penalty: float = 0.10
