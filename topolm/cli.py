from __future__ import annotations

import argparse
from .core import TopoLM
from .config import Config

DEMO = """
The cat sat on the mat.
The dog sat on the floor.
The attacker used CVE-2024-1234 to access the admin panel.
CYP3A4 inhibition increases drug exposure.
Clarithromycin inhibits CYP3A4.
Clarithromycin may increase simvastatin exposure.
"""

def build_demo_model():
    return TopoLM(Config()).fit(DEMO)

def main(argv=None):
    parser = argparse.ArgumentParser(prog="topolm")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("demo")
    p = sub.add_parser("predict"); p.add_argument("context")
    g = sub.add_parser("generate"); g.add_argument("prompt"); g.add_argument("--decoding", default="beam", choices=["beam", "nucleus", "greedy"])
    args = parser.parse_args(argv)
    model = build_demo_model()
    if args.cmd == "predict":
        for pred in model.distribution(args.context, 5):
            print(f"{pred.text}\t{pred.probability:.3f}\t{pred.score:.3f}")
    elif args.cmd == "generate":
        print(model.generate(args.prompt, decoding=args.decoding))
    else:
        print(model.generate("clarithromycin inhibits", decoding="beam"))

if __name__ == "__main__":
    main()
