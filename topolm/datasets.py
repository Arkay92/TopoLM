from __future__ import annotations

from typing import Iterable

try:
    from datasets import Dataset, load_dataset
except ImportError:  # pragma: no cover
    Dataset = None
    load_dataset = None


def _ensure_datasets_installed() -> None:
    if load_dataset is None:
        raise ImportError(
            "The Hugging Face datasets package is required for this feature. "
            "Install it with `pip install topolm[hf]` or `pip install datasets`."
        )


def _infer_text_field(dataset: "Dataset") -> str:
    if hasattr(dataset, "features"):
        for name, feature in dataset.features.items():
            if getattr(feature, "dtype", None) == "string":
                return name
    raise ValueError("Unable to infer a string text field from the dataset. Please pass text_field explicitly.")


def load_hf_dataset(
    dataset_name: str,
    split: str = "train",
    text_field: str | None = None,
    sample_size: int | None = None,
    shuffle_seed: int = 42,
) -> list[str]:
    _ensure_datasets_installed()
    ds = load_dataset(dataset_name, split=split)
    if text_field is None:
        text_field = _infer_text_field(ds)
    if text_field not in ds.column_names:
        raise ValueError(f"Dataset {dataset_name} does not contain a {text_field} field.")
    if sample_size is not None and sample_size > 0:
        ds = ds.shuffle(seed=shuffle_seed).select(range(min(sample_size, len(ds))))
    return [str(item) for item in ds[text_field]]


def hf_dataset_texts(
    dataset_name: str,
    split: str = "train",
    text_field: str | None = None,
    sample_size: int | None = None,
    shuffle_seed: int = 42,
) -> Iterable[str]:
    return load_hf_dataset(dataset_name, split=split, text_field=text_field, sample_size=sample_size, shuffle_seed=shuffle_seed)
