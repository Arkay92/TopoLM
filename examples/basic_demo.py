import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topolm import Config, TopoLM

corpus = """
The cat sat on the mat.
The dog sat on the floor.
The attacker used CVE-2024-1234 to access the admin panel.
CYP3A4 inhibition increases drug exposure.
Clarithromycin inhibits CYP3A4.
Clarithromycin may increase simvastatin exposure.
"""

model = TopoLM(Config()).fit(corpus)

print("Predictions for: clarithromycin inhibits")
for p in model.distribution("clarithromycin inhibits", 5):
    print(f"  {p.text:18s} prob={p.probability:.3f} score={p.score:.3f}")

print("\nBeam generation:")
print(model.generate("clarithromycin inhibits", decoding="beam"))

print("\nNucleus generation:")
print(model.generate("cyp3a4 inhibition", decoding="nucleus"))
