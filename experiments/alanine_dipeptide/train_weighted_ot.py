#!/usr/bin/env python3
"""Train weighted OT flow matching on alanine dipeptide."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common import base_parser
from experiments.alanine_dipeptide.training import train


if __name__ == "__main__":
    parser = base_parser(
        "Train weighted alanine-dipeptide OT flow matching", 1000, 5e-4
    )
    parser.set_defaults(
        output=ROOT / "runs/alanine_dipeptide/weighted_ot_checkpoint.pt"
    )
    train(parser.parse_args(), weighted=True)
