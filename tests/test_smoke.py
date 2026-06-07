from topolm import Config, TopoLM, load_hf_dataset

CORPUS = """
The cat sat on the mat.
The dog sat on the floor.
The attacker used CVE-2024-1234 to access the admin panel.
CYP3A4 inhibition increases drug exposure.
Clarithromycin inhibits CYP3A4.
Clarithromycin may increase simvastatin exposure.
"""

def test_predict_domain_terms():
    model = TopoLM(Config()).fit(CORPUS)
    preds = model.distribution("clarithromycin inhibits", 5)
    assert preds
    assert preds[0].text == "cyp3a4"

def test_generation_is_bounded_and_fluentish():
    model = TopoLM(Config()).fit(CORPUS)
    out = model.generate("cyp3a4 inhibition", decoding="beam", max_units=16)
    assert "CYP3A4" in out
    assert len(out.split()) <= 20

def test_save_load(tmp_path):
    model = TopoLM(Config()).fit(CORPUS)
    path = model.save(tmp_path / "model")
    loaded = TopoLM.load(path)
    assert loaded.distribution("clarithromycin inhibits", 3)[0].text == "cyp3a4"


def test_save_load_round_trip_state(tmp_path):
    model = TopoLM(Config()).fit(CORPUS)
    path = model.save(tmp_path / "model_state")
    loaded = TopoLM.load(path)
    assert loaded.mem.unit_counts == model.mem.unit_counts
    assert loaded.mem.phrase_counts == model.mem.phrase_counts
    assert loaded.mem.edge_counts == model.mem.edge_counts
    assert loaded.mem.domain_counts == model.mem.domain_counts


def test_fit_texts_helper():
    model = TopoLM(Config()).fit_texts([s.strip() for s in CORPUS.splitlines() if s.strip()])
    assert model.distribution("clarithromycin inhibits", 5)[0].text == "cyp3a4"


def test_hf_dataset_loader_simple(monkeypatch):
    import pytest
    datasets = pytest.importorskip("datasets")
    from datasets import Dataset

    def fake_load_dataset(name, split="train"):
        return Dataset.from_dict({"text": ["Topologist test text."]})

    monkeypatch.setattr("topolm.datasets.load_dataset", fake_load_dataset)
    texts = load_hf_dataset("testset", split="train", text_field="text", sample_size=1)
    assert texts == ["Topologist test text."]
